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
import kornia

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

import envs
from envs.ffmp.robot import RobotPosition, RobotVelocity, RobotState, RobotAction
from envs.ffmp.ffmp import FFMP



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state_m', 'state_g', 'state_v', 'action', 'observe_m', 'observe_g', 'observe_v', 'reward'))


##### ROS #####

odom = Odometry()
bridge = CvBridge()
tensor_map = [] * 2
flow_callback_flag = False
occupancy_callback_flag = False
odom_callback_flag = False
posearray_callback_flag = False

class ROSNode():
    def __init__(self):
        self.sub_flow = rospy.Subscriber("/bev/flow_image", Image, self.flow_image_callback)
        self.sub_occupancy = rospy.Subscriber("/occupancy_image", Image, self.occupancy_image_callback)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.cmd_vel_callback)
        self.sub_start_goal = rospy.Subscriber("/start_goal", PoseArray, self.pose_array_callback)
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel/train", Twist, queue_size=1)
        self.odom = Odometry()
        self.global_start = Pose()
        self.global_goal = Pose()

    def occupancy_image_callback(self, msg):
        print("occupancy_image_callback")
        cv_occupancy_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        tensor_occupancy_image = kornia.image_to_tensor(cv_occupancy_image, keepdim=True).float()
        tensor_map[0] = tensor_occupancy_image
        occupancy_callback_flag = True

    def flow_image_callback(self, msg):
        print("image_callback")
        cv_flow_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_flow_image = cv2.cvtColor(cv_flow_image, cv2.COLOR_BGR2RGB)
        tensor_flow_image = kornia.image_to_tensor(cv_flow_image, keepdim=True).float()
        tensor_map[1] = tensor_flow_image
        flow_callback_flag = True

    def odom_callback(self, msg):
        print("odom_callback")
        self.odom = msg
        odom_callback_flag = True

    def pose_array_callback(self, msg):
        print("pose_array_callback")
        self.global_start.pose = msg[0].poses # or [1]
        self.global_goal.pose = msg[1].poses # or [0]
        posearray_callback_flag = True

    def cmd_vel_publisher(self, linear_v, angular_v):
        print("pub")
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_v
        cmd_vel.linear.y = 0.0
        cmd_vel.lineat.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = angular_v
        self.pub.publish(cmd_vel)

    def robot_position_extractor(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        roll, pitch, yaw = euler_from_quaternion(odom.pose.pose.orientation)
        robot_pose = RobotPose(x, y, yaw)
        return robot_pose

    def pi_to_pi(self, angle):
        while angle>=pi:
            angle=angle-2*pi
        while angle<=-pi:
            angle=angle+2*pi
        return angle

    def relative_goal_calculator(self, robot_pose):
        relative_x = self.global_goal.pose.position.x - robot_pose.x
        relative_y = self.global_goal.pose.position.y - robot_pose.y
        relative_dist = math.sqrt(relative_x * relative_x + relative_y * relative_y)
        relative_direction = math.atan2(relative_y, relative_x)
        relative_orientation = self.pi_to_pi(relative_direction - robot_pose.yaw)
        return np.array([relative_dist, relative_orientation])

    def robot_velocity_calculator(self, robot_pose, is_first):
        if is_first:
            pre_robot_pose = copy.deepcopy(robot_pose)
        linear_v = math.sqrt(math.pow(robot_pose.x - pre_robot_pose.x, 2) + math.pow(robot_pose.y - pre_robot_pose.y, 2))
        angular_v = self.pi_to_pi(robot_pose.yaw - pre_robot_pose.yaw)
        return np.array([linear_v, angular_v])


##### DDQN #####
ENV = 'FFMP-v0'
GAMMA = 0.99 # discount factor
MAX_STEPS = 10000
NUM_EPISODES = 500

# params for Brain class
BATCH_SIZE = 1024 # minibatch size
CAPACITY = 200000 # replay buffer size
INPUT_CHANNELS = 12 #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)
NUM_ACTIONS = 28
LEARNING_RATE = 0.0005 # learning rate


class ReplayMemory(object):
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward):
        """Saves a transition."""
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
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_ea = nn.Linear(512, outputs) # Elements A(s,a) that depend on your actions.
        self.fc4_ev = nn.Linear(512, 1) # Elements V(s) that are determined only by the state.

    def forward(self, state_m, state_g, state_v):
        x_m = F.relu(self.conv1(concat_maps))
        x_m = F.relu(self.conv2(x_m))
        x_m = F.relu(self.conv3(x_m))

        x_gv_ = torch.cat((state_g, state_v, 0)
        x_gv_ = F.relu(self.fc1(x_gv_))
        convw = x_gv_.shape[0]
        convh = convw
        x_gv = torch.empty(convw, convh)
        
        for i in range(64):
            tile_gv = torch.full((convw, convh), fill_value=x_gv_[i])
            x_gv = x_gv.stack(tile_gv, dim=0)

        x_pls = x_m + x_gv

        x_pls = F.relu(self.conv3(x))
        x_pls = F.relu(self.conv3(x))
        x_pls = F.relu(self.conv3(x))

        x = torch.flatten(x_pls)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        adv = self.fc4_ea(x)
        val = self.fc4_ev(x)

        return adv + val - adv.mean(1, keepdim=True).expand(-1, adv.size(1))


class Brain:
    def __init__(self):
        self.num_actions = NUM_ACTIONS
        self.memory = ReplayMemory(CAPACITY)
        self.main_q_network = Network(INPUT_CHANNELS, NUM_ACTIONS)
        self.target_q_network = Network(INPUT_CHANNELS, NUM_ACTIONS)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=LEARNING_RATE)

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

    def memorize(self, state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward)
        self.brain.memory.push(state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.agent = Agent()
        self.map_memory = []

    def make_concat_map(self, occupancy_map, flow_map):
        # occupancy_map.size() = torch.Size([H, W])
        # flow_map.size() = torch.Size([3, H, W])
        occupancy_map = tf.expand_dims(occupancy_map, 0)
        # occupancy_map.size() = torch.Size([1, H, W])
        concat_map = torch.cat((occupancy_map, flow_map), 0)
        #concat_map.size() = torch.Size([4, H, W])    
        return concat_map
    
    def make_temporal_maps(concat_map, is_first):
        if is_first:
            self.map_memory.clear()
            for i in range(3):
                self.map_memory.append(concat_map)
        else:
            self.map_memory.append(concat_map)
            del self.map_memory[0]
        
        tempral_maps = torch.cat(self.map_memory, 0)
        # tempral_maps.size() = torch.Size([12, H, W])
        return tempral_maps


def main():
    rospy.init_node('train', anonymous=True)
    ros = ROSNode()

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

    while not rospy.is_shutdown():
        if flow_callback_flag and occupancy_callback_flag and odom_callback_flag and posearray_callback_flag:

            robot_pose = ros.robot_position_extractor() # numpy
            relative_goal = ros.relative_goal_calculator(robot_pose) # numpy
            tmp_relative_goal = copy.deepcopy(relative_goal) # numpy
            velocity = ros.robot_velocity_calculator(robot_pose, is_first) # numpy
            occpancy_map = tensor_map[0] # tensor
            flow_map = tensor_map[1] #tensor

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

            action = train_env.agent.get_action(state_m, state_g, state_v, episode)
            action_id = action.item()

            tmp_tensor_occupancy = occupancy_map.clone()
            numpy_occupancy = tmp_tensor_occupancy.to('cpu').detach().numpy()
            reward, is_done = train_env.env.rewarder(numpy_occupancy, relative_goal)

            is_first = False
            step += 1

            train_env.agent.memorize(state_m, state_g, state_v, action, observe_m, observe_g, observe_v, reward)

            if step == MAX_STEPS:
                is_done = True

            if is_done:
                episode += 1
                step = 0
                state_m = None
                state_g = None
                state_v = None

                if(episode % 2 == 0):
                    self.agent.update_target_q_function()

                ros.cmd_vel_publisher(0.0, 0.0)
                is_first = True
            else:
                linear_v = train_env.agent.action.commander[action_id].linear_v
                angular_v = train_env.agent.action.commander[action_id].angular_v
                ros.cmd_vel_publisher(linear_v, angular_v)

                state_m = observe_m
                state_g = observe_g
                state_v = observe_v

            flow_callback_flag = False
            occupancy_callback_flag = False
            odom_callback_flag = False

        rospy.spin()



if __name__ == '__main__':
    main()
