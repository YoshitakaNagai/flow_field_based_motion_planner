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
env = gym.make('FFMP-v0')
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


##### ROS #####

odom = Odometry()
bridge = CvBridge()
global_start_pose = Pose()
global_goal_pose = Pose()
relative_goal = np.array([0.0, 0.0])
tensor_maps = [] * 2
temporal_tensor_maps = [] * 3

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
        tensor_flow_image = kornia.image_to_tensor(cv_flow_image, keepdim=False).float()
        tensor_maps[0] = tensor_flow_image

    def occupancy_image_callback(self, msg):
        print("occupancy_image_callback")
        cv_occupancy_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        tensor_occupancy_image = kornia.image_to_tensor(cv_occupancy_image, keepdim=False).float()
        tensor_maps[1] = tensor_occupancy_image

    def odom_callback(self, msg):
        print("odom_callback")
        odom = msg

    def pose_array_callback(self, msg):
        print("pose_array_callback")
        global_start.pose = msg[0].poses # or [1]
        global_goal.pose = msg[1].poses # or [0]

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
        self.position = RobotPosition(0.0, 0.0, 0.0)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, h, w, input_channels, outputs=28):
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

    def forward(self, local_map, goal, velocity):
        x_lm = F.relu(self.conv1(local_map))
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

n_actions = 28

policy_net = Network(screen_height, screen_width, n_actions).to(device) # for learning
target_net = Network(screen_height, screen_width, n_actions).to(device) # for evaluating
target_net.load_state_dict(policy_net.state_dict()) # loading Network model
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # eps_threshold = 0.5 * (1 / (steps_done + 1))
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        # Return a int number from 0 to 27 if n_actions=28

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def reset():


def optimization_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def training_loop():
    num_episodes = 50
    for i_episode in range(num_episodes):
        # [1] Initialize the environment and state

        for t in count():
            # [1-1] Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # [1-2] Observe new state

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # [1-3] Store the transition in memory
            memory.push(state, action, next_state, reward)

            # [1-4] Move to the next state
            state = next_state

            # [1-5] Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        # [2] Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())









NUM_ACTIONS = 28

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)
        
        flow_map = tensor_maps[0]
        h = flow_map.size()[0]
        w = flow_map.size()[1]
        input_channels = 12 #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)
        self.main_q_network = Network(h, w, input_channels, num_actions)






























def main():
    rospy.init_node('train', anonymous=True)

    ros = ROSNode()

    while not rospy.is_shutdown():
        rospy.spin()



if __name__ == '__main__':
    main()
