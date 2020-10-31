#!/usr/bin/env python3
import gym
import math
import random
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
# env = gym.make('FFMP-v0')
Transition = namedtuple('Transition', ('pre_occupancy_map', 'pre_flow_map', 'pre_relative_goal', 'pre_action', 'action', 'occupancy_maps', 'flow_maps', 'reward'))


##### ROS #####

odom = Odometry()
bridge = CvBridge()
global_start_pose = Pose()
global_goal_pose = Pose()
relative_goal = np.array([0.0, 0.0])
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

    def flow_image_callback(self, msg):
        print("image_callback")
        cv_flow_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_flow_image = cv2.cvtColor(cv_flow_image, cv2.COLOR_BGR2RGB)
        tensor_flow_image = kornia.image_to_tensor(cv_flow_image, keepdim=True).float()
        tensor_map[0] = tensor_flow_image
        flow_callback_flag = True

    def occupancy_image_callback(self, msg):
        print("occupancy_image_callback")
        cv_occupancy_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        tensor_occupancy_image = kornia.image_to_tensor(cv_occupancy_image, keepdim=True).float()
        tensor_map[1] = tensor_occupancy_image
        occupancy_callback_flag = True

    def odom_callback(self, msg):
        print("odom_callback")
        odom = msg
        odom_callback_flag = True

    def pose_array_callback(self, msg):
        print("pose_array_callback")
        global_start.pose = msg[0].poses # or [1]
        global_goal.pose = msg[1].poses # or [0]
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


##### DDQN #####

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0
        # self.position = RobotPosition(0.0, 0.0, 0.0)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # self.memory[self.position] = Transition(*args)
        self.index = (self.index + 1) % self.capacity
        # self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, h, w, input_channels, outputs):
        print('[deepRL]  DQN::__init__()')
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_ea = nn.Linear(512, outputs) # Elements A(s,a) that depend on your actions.
        self.fc4_ev = nn.Linear(512, 1) # Elements V(s) that are determined only by the state.

    def forward(self, concat_maps, goal, velocity):

        x_lm = F.relu(self.conv1(concat_maps))
        x_lm = F.relu(self.conv2(x_lm))
        x_lm = F.relu(self.conv3(x_lm))

        x_gv_ = torch([goal, velocity])
        x_gv_ = F.relu(self.fc1(x_gv_))
        convw = x_gv.shape[0]
        convh = x_gv.shape[1]
        x_gv = torch.empty(convw, convh)
        
        for i in range(64):
            tile = torch.full((convw, convh), fill_value=x_gv_[i])
            x_gv = x_gv.stack(tile, dim=0)

        x_pls = x_lm + x_gv

        x_pls = F.relu(self.conv3(x))
        x_pls = F.relu(self.conv3(x))
        x_pls = F.relu(self.conv3(x))

        x = torch.flatten(x_pls)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        adv = self.fc4_ea(x)
        val = self.fc4_ev(x)

        return adv + val - adv.mean(1, keepdim=True).expand(-1, adv.size(1))


def robot_position_extractor(self):
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y
    roll, pitch, yaw = euler_from_quaternion(odom.pose.pose.orientation)
    robot_pose = RobotPosition(x, y, yaw)
    
    return robot_pose


def relative_goal_calculator(self, robot_pose):
    relative_x = global_goal.pose.position.x - robot_pose.x
    relative_y = global_goal.pose.position.y - robot_pose.y
    relative_dist = math.sqrt(relative_x * relative_x + relative_y * relative_y)
    relative_direction = math.atan2(relative_y, relative_x)
    relative_orientation = relative_direction - robot_pose.yaw

    return np.array([relative_dist, relative_orientation])


env.reset()

ENV = 'FFMP-v0'
LEARNING_RATE = 0.0005 # learning rate
GAMMA = 0.99 # discount factor
REPLAY_BUFFER_SIZE = 200000 # replay buffer size
BATCH_SIZE = 1024 # minibatch size
IMAGE_SIZE = [60, 60]
INITIAL_EXPLORATION = 1.0
EPISODE_LENGTH = 300
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10 # the number to decide updating step
EPS_START = 0.9 # the param for eps-greedy
EPS_END = 0.05 # the param for eps-greedy
EPS_DECAY = 300 # the param for eps-greedy

MAX_STEPS = 10000

NUM_ACTIONS = 28

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        input_channels = 12 #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)

        # h = occupancy_map.size()[0]
        # w = occupancy_map.size()[1]
        h = 60
        w = 60

        self.main_q_network = Network(h, w, input_channels, num_actions)
        self.target_q_network = Network(h, w, input_channels, num_actions)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=LEARNING_RATE)

    def replay(self):

        # [1] check memory size
        if len(self.memory) < BATCH_SIZE:
            return

        # [2] make mini batch
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

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
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

    def make_minibatch(self):
        # [2-1] extract mini batch data from memory
        transitions = self.memory.sample(BATCH_SIZE)

        # [2-2] convert changable_nums like below
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # [2-3] convert elements
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        # [3] get Q(s_t, a_t) value
        # [3-1] change network to evaluation mode
        self.main_q_network.eval()
        self.target_q_network.eval()

        # [3-2] get Q(s_t, a_t) from main_q_network
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # [3-3] get max{Q(s_t+1, a)}
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # [3-4] get Q(s_t, a_t) from the equision of Q Learnning
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        # [4] update of connected param
        
        # [4-1] change network to trainning mode
        self.main_q_network.train()
        
        # [4-2] calc loss
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqeeze(1))

        # [4-3] update of connected params
        self.optimizer.zero_grad() # reset grad
        loss.backward() # calc back propagate
        self.optimizer.step() # update connected params

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state_m, state_g, state_v, episode):
        action = self.brain.decide_action(state_m, state_g, state_v, episode)
        return action

    def memorize(self, pre_occupancy_maps, pre_flow_maps, pre_relative_goal, pre_action, action, occupancy_maps, flow_maps, relative_goal, reward)
        self.brain.memory.push(pre_occupancy_map, pre_flow_map, pre_relative_goal, pre_action, action, occupancy_maps, flow_maps, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        # num_states = self.env.observation_space.shape[0] # It's supposed to be 3; local_map, relative_goal, velocity
        num_states = 3 # It's supposed to be 3; local_map, relative_goal, velocity
        num_actions = self.env.action_space.n # = 28
        self.agent = Agent(num_states, num_actions)
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


    def run(self, occupancy_map, flow_map, observation_gv, episode, is_first):

        numpy_occupancy = occupancy_map.clone().numpy()
        observation_m = self.make_concat_map(occupancy_map, flow_map)
        state_m = self.make_temporal_maps(observation_m, is_first)
        numpy_g = observation_gv[0,:]
        state_g = torch.from_numpy(numpy_g).type(torch.FloatTensor)
        numpy_v = observation_gv[1,:]
        state_v = torch.from_numpy(numpy_v).type(torch.FloatTensor)
        
        action = self.agent.get_action(state_m, state_g, state_v, episode)
        # numpy_action = action.clone().numpy()

        # observation_gv_next, reward, is_done = self.env.step(numpy_occupancy, numpy_g, numpy_action)
        reward, is_done = self.env.step(numpy_occupancy, numpy_g)

        # return observation_gv_next, reward, is_done
        return action, reward, is_done


def main():
    rospy.init_node('train', anonymous=True)

    ros = ROSNode()
    train_env = Environment()
    episode = 0
    step = 0
    is_first = True
    done = True
    
    action = None
    pre_relative_goal = None
    pre_action = None
    pre_tensor_maps = None

    numpy_next_gv = None
    state_gv = None
    state_next_gv = None


    while not rospy.is_shutdown():
        if flow_callback_flag and occupancy_callback_flag and odom_callback_flag and posearray_callback_flag:

            robot_pose = robot_position_extractor()
            relative_goal = relative_goal_calculator(robot_pose)

            if is_first:
                action, reward = train_env.env.reset()
                pre_relative_goal = relative_goal
                pre_tensor_map = tensor_map

            # observation_next_gv, reward, is_done = train_env.run(tensor_map[0], tensor_map[1], state_gv, episode, is_first)
            action, reward, is_done = train_env.run(tensor_map[0], tensor_map[1], state_gv, episode, is_first)

            is_first = False
            step += 1

            if is_done:
                state_next_gv = None
                episode_n_list = np.hstack(episode_n_list[1:], step + 1)

                if step == MAX_STEPS:
                    is_done = True
            else:
                state_next_gv = observation_next_gv
                state_next_gv = torch.from_numpy(state_next_gv).type(torch.FloatTensor)

            train_env.agent.memorize(pre_tensor_map[0], pre_tensor_map[1], pre_relative_goal, pre_action, action, tensor_map[0], tensor_map[1], relative_goal, reward)

            if is_done:
                episode += 1
                step = 0
                is_first = True
                is_done = True

            linear_v = action.cmd.linear_v
            angular_v = action.cmd.angular_v
            ros.cmd_vel_publisher(linear_v, angular_v)

            pre_action = action
            pre_relative_goal = relative_goal
            pre_tensor_map = tensor_map

        rospy.spin()



if __name__ == '__main__':
    main()
