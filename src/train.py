#!/usr/bin/env python3
import gym
import math
import random
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import kornia

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

import cv2
from cv_bridge import CvBridge

import gym_ffmp
from gym_ffmp.envs.robot.config import RobotPose, RobotVelocity, RobotState, RobotAction
from gym_ffmp.envs.ffmp import FFMP



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state_m', 'state_g', 'state_v', 'action', 'observe_m', 'observe_g', 'observe_v', 'reward'))
model_path = '../model/model.pt'


##### ROS #####

maps = [None] * 3 # [0]:occupancy with robot, [1]:flow, [2]:occupancy without robot

IMAGE_SIZE = 40

class ROSNode():
    def __init__(self):
        self.sub_flow = rospy.Subscriber("/bev/flow_image", Image, self.flow_image_callback)
        self.sub_occupancy = rospy.Subscriber("/bev/occupancy_image", Image, self.occupancy_image_callback)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.sub_start_goal = rospy.Subscriber("/start_goal", PoseArray, self.pose_array_callback)
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel/train", Twist, queue_size=1)
        self.pub_done_flag = rospy.Publisher("/train/episode", Bool, queue_size=1)

        self.odom = Odometry()
        self.global_start = Pose()
        self.global_goal = Pose()
        self.bridge = CvBridge()
        self.done_flag = Bool()
        self.pre_robot_pose = RobotPose(0.0, 0.0, 0.0)
        self.map_grid_size = 0.1
        self.map_range = 5.0
        self.robot_rsize = 0.1

        self.flow_callback_flag = False
        self.occupancy_callback_flag = False
        self.odom_callback_flag = False
        self.posearray_callback_flag = False

    def occupancy_image_callback(self, msg):
        # print("occupancy_image_callback")
        cv_occupancy_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        maps[2] = cv_occupancy_image.copy()

        # tensor_occupancy_image = kornia.image_to_tensor(cv_occupancy_image, keepdim=True).float()
        tensor_occupancy_image = torch.from_numpy(cv_occupancy_image)

        for i in range(len(cv_occupancy_image[0])):
            for j in range(len(cv_occupancy_image[1])):
                if math.sqrt(math.pow(i * self.map_grid_size - 0.5 * self.map_range, 2) + math.pow(j * self.map_grid_size - 0.5 * self.map_range, 2)) <= self.robot_rsize:
                    # cv_occupancy_image[i, j] = 1.0
                    tensor_occupancy_image[i, j] = 1

        maps[0] = tensor_occupancy_image
        self.occupancy_callback_flag = True

    def flow_image_callback(self, msg):
        # print("flow_image_callback")
        cv_flow_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_flow_image = cv2.cvtColor(cv_flow_image, cv2.COLOR_BGR2RGB)
        tensor_flow_image = kornia.image_to_tensor(cv_flow_image, keepdim=True).float()
        # tensor_flow_image = torch.from_numpy(cv_flow_image)
        maps[1] = tensor_flow_image
        self.flow_callback_flag = True

    def odom_callback(self, msg):
        # print("odom_callback")
        self.odom = msg
        self.odom_callback_flag = True

    def pose_array_callback(self, msg):
        # print("pose_array_callback")
        self.global_start.pose = msg[0].poses
        self.global_goal.pose = msg[1].poses
        self.posearray_callback_flag = True

    def cmd_vel_publisher(self, linear_v, angular_v):
        # print("pub cmd_vel")
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_v
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = angular_v
        self.pub_cmd_vel.publish(cmd_vel)

    def done_flag_publisher(self, is_done):
        # print("pub done_flag")
        self.done_flag = is_done
        self.pub_done_flag.publish(self.done_flag)

    def robot_position_extractor(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        orientation_q = self.odom.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        robot_pose = RobotPose(x, y, yaw)
        return robot_pose

    def pi_to_pi(self, angle):
        while angle >= math.pi:
            angle = angle - (2 * math.pi)
        while angle <= -math.pi:
            angle = angle + 2 * math.pi
        return angle

    def relative_goal_calculator(self, robot_pose):
        relative_x = self.global_goal.position.x - robot_pose.x
        relative_y = self.global_goal.position.y - robot_pose.y
        relative_dist = math.sqrt(relative_x * relative_x + relative_y * relative_y)
        relative_direction = math.atan2(relative_y, relative_x)
        relative_orientation = self.pi_to_pi(relative_direction - robot_pose.yaw)
        return np.array([relative_dist, relative_orientation])

    def robot_velocity_calculator(self, robot_pose, is_first):
        if is_first:
            self.pre_robot_pose = copy.deepcopy(robot_pose)
        linear_v = math.sqrt(math.pow(robot_pose.x - self.pre_robot_pose.x, 2) + math.pow(robot_pose.y - self.pre_robot_pose.y, 2))
        angular_v = self.pi_to_pi(robot_pose.yaw - self.pre_robot_pose.yaw)
        self.pre_robot_pose = copy.deepcopy(robot_pose)
        return np.array([linear_v, angular_v])
    


##### DDQN #####
ENV = 'FFMP-v0'
GAMMA = 0.99 # discount factor
MAX_STEPS = 10000
NUM_EPISODES = 500
LOG_DIR = "./logs/experiment1"

# params for Brain class
BATCH_SIZE = 1024 # minibatch size
CAPACITY = 200000 # replay buffer size
INPUT_CHANNELS = 12 #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)
NUM_ACTIONS = 28
LEARNING_RATE = 0.0005 # learning rate
LOSS_THRESHOLD = 0.1 # threshold of loss
LOSS_MEMORY_CAPACITY = 10

class ReplayMemory(object):
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, input_channels, outputs):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(6400, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_ea = nn.Linear(512, outputs) # Elements A(s,a) that depend on your actions.
        self.fc4_ev = nn.Linear(512, 1) # Elements V(s) that are determined only by the state.

    def forward(self, state_m, state_g, state_v):
        x_m = F.relu(self.conv1(state_m))
        # print("conv1 -> x_m.size() = ", x_m.size())
        x_m = F.relu(self.conv2(x_m))
        # print("conv2 -> x_m.size() = ", x_m.size())
        x_m = F.relu(self.conv3(x_m))
        # print("conv3 -> x_m.size() = ", x_m.size())

        x_gv_ = torch.cat((state_g, state_v), 0)
        x_gv_ = F.relu(self.fc1(x_gv_))
        convw = x_gv_.shape[0]
        convh = convw
        x_gv = torch.empty(convw, convh)
        
        for i in range(16):
            tile_gv = torch.full((convw, convh), fill_value=x_gv_[i])
            x_gv = torch.stack(tuple(tile_gv), 0)
        
        # print("x_gv.size() = ", x_gv.size())

        x_m = x_m.to('cpu')
        x_gv = x_gv.to('cpu')
        x_pls = x_m + x_gv
        x_pls = x_pls.to(device)

        x_pls = F.relu(self.conv3(x_pls))
        x_pls = F.relu(self.conv3(x_pls))
        x_pls = F.relu(self.conv3(x_pls))
        # print("x_pls.size() = ", x_pls.size())

        x = torch.flatten(x_pls)
        # print("x.size() = ", x.size())

        x = F.relu(self.fc2(x))
        # print("fc2 -> x.size() = ", x.size())
        x = F.relu(self.fc3(x))
        # print("fc3 -> x.size() = ", x.size())
        adv = self.fc4_ea(x)
        val = self.fc4_ev(x)
        adv = torch.unsqueeze(adv, 0)
        # print("adv.size() = ", adv.size())
        val = torch.unsqueeze(val, 0)
        # print("val.size() = ", val.size())
        adv = adv.to('cpu')
        val = val.to('cpu')

        output = adv + val - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        output = output.to(device)
        print("output.size()", output.size())

        return output


class Brain:
    def __init__(self):
        self.num_actions = NUM_ACTIONS
        self.memory = ReplayMemory()
        self.main_q_network = Network(INPUT_CHANNELS, NUM_ACTIONS).to(device)
        self.target_q_network = Network(INPUT_CHANNELS, NUM_ACTIONS).to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=LEARNING_RATE)
        self.loss = None
        self.step = 0

    def replay(self):

        # [1] check memory size
        if len(self.memory) < BATCH_SIZE:
            return

        # [2] make mini batch
        self.batch, self.state_m_batch, self.state_g_batch, self.state_v_batch, self.action_batch, self.reward_batch, self.non_final_next_states_m, self.non_final_next_states_g, self.non_final_next_states_v = self.make_minibatch()

        # [3] calc Q(s_t, a_t) value
        self.expected_state_action_values = self.get_expected_state_action_values()

        # [4] update connected params
        self.update_main_q_network()


    def decide_action(self, state_m, state_g, state_v, episode):
        # epsilon-greedy method
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval() # change network to evaluation mode
            with torch.no_grad():
                action = self.main_q_network(state_m, state_g, state_v).max(1)[1].view(1, 1)
        else:
            action = torch.cuda.LongTensor([[random.randrange(self.num_actions)]])

        return action

    def make_minibatch(self):
        # [2-1] extract mini batch data from memory
        transitions = self.memory.sample(BATCH_SIZE)

        # [2-2] convert changable_nums like below
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # [2-3] convert elements
        state_m_batch = torch.cat(batch.state_m)
        state_g_batch = torch.cat(batch.state_g)
        state_v_batch = torch.cat(batch.state_v)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states_m = torch.cat([s for s in batch.observe_m if s is not None])
        non_final_next_states_g = torch.cat([s for s in batch.observe_g if s is not None])
        non_final_next_states_v = torch.cat([s for s in batch.observe_v if s is not None])

        return batch, state_m_batch, state_g_batch, state_v_batch, action_batch, reward_batch, non_final_next_states_m, non_final_next_states_g, non_final_next_states_v

    def get_expected_state_action_values(self):
        # [3] get Q(s_t, a_t) value
        # [3-1] change network to evaluation mode
        self.main_q_network.eval()
        self.target_q_network.eval()

        # [3-2] get Q(s_t, a_t) from main_q_network
        self.state_action_values = self.main_q_network(self.state_m_batch, self.state_g_batch, self.state_v_batch).gather(1, self.action_batch)

        # [3-3] get max{Q(s_t+1, a)}
        non_final_mask = torch.cuda.ByteTensor(tuple(map(lambda s: s is not None, {self.batch.observe_m, self.batch.observe_g, self.batch.observe_v})))
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.cuda.LongTensor)
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states_m, self.non_final_next_states_g, self.non_final_next_states_v).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states_m, self.non_final_next_states_g, self.non_final_next_states_v).gather(1, a_m_non_final_next_states).detach().squeeze()

        # [3-4] get Q(s_t, a_t) from the equision of Q Learnning
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        # [4] update of connected param
        
        # [4-1] change network to trainning mode
        self.main_q_network.train()
        
        # [4-2] calc loss
        # loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqeeze(1))
        loss = nn.MSELoss(self.state_action_values, self.expected_state_action_values.unsqeeze(1))
        self.loss = loss

        # [4-3] update of connected params
        self.optimizer.zero_grad() # reset grad
        loss.backward() # calc back propagate
        self.optimizer.step() # update connected params

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())



class Agent:
    def __init__(self):
        self.brain = Brain()
        self.action = RobotAction()

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state_m, state_g, state_v, episode):
        action = self.brain.decide_action(state_m, state_g, state_v, episode)
        return action

    def memorize(self, state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward):
        self.brain.memory.push(state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.agent = Agent()
        self.map_memory = []
        self.loss_memory = []
        self.loss_memory_capacity = LOSS_MEMORY_CAPACITY
        self.loss_ave = None
        self.loss_convergence = False

    def make_concat_map(self, occupancy_map, flow_map):
        # occupancy_map.size() = torch.Size([H, W])
        # flow_map.size() = torch.Size([3, H, W])

        occupancy_map = torch.unsqueeze(occupancy_map, 0)
        # print("occupancy_map.size() = ", occupancy_map.size())
        # print("flow_map.size() = ", flow_map.size())
        
        # occupancy_map.size() = torch.Size([1, H, W])
        repeat_vals = (1, 40, 40)
        concat_map = torch.cat((flow_map, occupancy_map.expand(*repeat_vals)), 0)
        # print("concat_map.size() = ", concat_map.size())
        # concat_map = torch.stack((flow_map, occupancy_map), 2)
        #concat_map.size() = torch.Size([4, H, W])    
        # concat_map = torch.unsqueeze(concat_map, 0)
        
        return concat_map
    
    def make_temporal_maps(self, concat_map, is_first):
        if is_first:
            self.map_memory.clear()
            for i in range(3):
                self.map_memory.append(concat_map)
        else:
            self.map_memory.append(concat_map)
            del self.map_memory[0]
        
        temporal_maps = torch.cat(self.map_memory, 0)
        # print("temporal_maps.size() = ", temporal_maps.size())
        # temporal_maps.size() = torch.Size([12, H, W])
        return temporal_maps

    def check_convergence(self):
        if len(self.loss_memory) > self.loss_memory_capacity:
            del self.loss_memory[0]
        
        self.loss_memory.append(self.agent.brain.loss)
        self.loss_ave = sum(self.loss_memory) / len(self.loss_memory)

        if self.loss_ave < LOSS_THRESHOLD:
            self.loss_convergence = True

class UseTensorBord:
    def __init__(self):
        self.writer = SummaryWriter(log_dir=LOG_DIR)
        # self.log_loss = []


def main():
    rospy.init_node('train', anonymous=True)
    ros = ROSNode()
    r = rospy.Rate(100)

    train_env = Environment()

    episode = 0
    step = 0
    is_first = True
    is_done = True
    state_m = None
    state_g = None
    state_v = None
    action = None
    action_id = 3
    is_complete = False

    tensor_board = UseTensorBord()

    print("ros : start!")
    while not rospy.is_shutdown():
        # if ros.flow_callback_flag and ros.occupancy_callback_flag and ros.odom_callback_flag and ros.posearray_callback_flag:
        if ros.flow_callback_flag and ros.occupancy_callback_flag and ros.odom_callback_flag:

            # print("ros flags : OK")
            robot_pose = ros.robot_position_extractor() # numpy
            relative_goal = ros.relative_goal_calculator(robot_pose) # numpy
            tmp_relative_goal = copy.deepcopy(relative_goal) # numpy
            velocity = ros.robot_velocity_calculator(robot_pose, is_first) # numpy
            occupancy_map = maps[0] # tensor
            flow_map = maps[1] #tensor

            concat_map = train_env.make_concat_map(occupancy_map, flow_map)
            observe_m = train_env.make_temporal_maps(concat_map, is_first)
            observe_g = torch.from_numpy(tmp_relative_goal).type(torch.FloatTensor)
            observe_v = torch.from_numpy(velocity).type(torch.FloatTensor)

            if is_first:
                reward = 0
                state_m = observe_m
                state_g = observe_g
                state_v = observe_v
                action_id = 3
            
            state_m = torch.unsqueeze(state_m, 0)
            state_m = state_m.to(device)
            state_g = state_g.to(device)
            state_v = state_v.to(device)

            action = train_env.agent.get_action(state_m, state_g, state_v, episode)
            action_id = action.item()

            tmp_tensor_occupancy = occupancy_map.clone()
            numpy_occupancy = tmp_tensor_occupancy.to('cpu').detach().numpy()
            reward, is_done = train_env.env.rewarder(numpy_occupancy, relative_goal, is_first)

            is_first = False
            train_env.agent.brain.step += 1

            train_env.agent.memorize(state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward)

            ros.flow_callback_flag = False
            ros.occupancy_callback_flag = False
            ros.odom_callback_flag = False

            if step == MAX_STEPS:
                is_done = True
            
            
            if is_done:
                print("is_done : ", is_done)
                print("episode", episode, ": loss = ", train_env.agent.brain.loss)
                tensor_board.writer.add_scalar("ours", train_env.agent.brain.loss, episode)

                episode += 1
                step = 0
                state_m = None
                state_g = None
                state_v = None

                if(episode % 2 == 0):
                    self.agent.update_target_q_function()

                if train_env.loss_convergence:
                    torch.save(train_env.agent.brain.main_q_network.state_dict(), model_path)
                    is_complete = True

                ros.cmd_vel_publisher(0.0, 0.0)
                is_first = True
            else:
                print("step:",step , ", loss:",train_env.agent.brain.loss)
                # output_velocity = train_env.agent.action.commander()
                linear_v = train_env.agent.action.commander(action_id).linear_v
                angular_v = train_env.agent.action.commander(action_id).angular_v
                ros.cmd_vel_publisher(linear_v, angular_v)
                ros.done_flag_publisher(is_done)

                state_m = observe_m
                state_g = observe_g
                state_v = observe_v

                step += 1

        if is_complete:
            print("complete")
            tensor_board.writer.close()
            break

        # rospy.spin()
        r.sleep()

if __name__ == '__main__':
    main()
