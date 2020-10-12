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
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

odom = Odometry()
#flow_image = Image()
bridge = CvBridge()

class ROSNode():
    def __init__(self):
        self.sub = rospy.Subscriber("/bev/flow_image", Image, self.image_callback)
        self.sub = rospy.Subscriber("/occupancy_grid", Image, self.occupancy_grid_callback)
        self.sub = rospy.Subscriber("/odom", Odometry, self.cmd_vel_callback)
        self.pub = rospy.Publisher("/cmd_vel/train", Twist, queue_size=1)

    def image_callback(self, msg):
        print("image_callback")
        cv_flow_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_flow_image = cv2.cvtColor(cv_flow_image, cv2.COLOR_BGR2RGB)
        tensor_flow_image = kornia.image_to_tensor(cv_flow_image, keepdim=False).float()
    
    def occupancy_grid_callback(self, msg):
        print("occupancy_grid_callback")

    def odom_callback(self, msg):
        print("odom_callback")
        odom = msg

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





class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = RobotPosition()

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


class DQN(nn.Module):
    def __init__(self, h, w, input_channels, outputs=28):
        print('[deepRL]  DQN::__init__()')
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_ea = nn.Linear(512, outputs) # Elements A(s,a) that depend on your actions.
        self.fc4_ev = nn.Linear(512, 1) # Elements V(s) that are determined only by the state.

    def conv2d_size_out(size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride  + 1

    # def calc_convw(w):
    #     return conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
    # def calc_convh(h):
    #     return conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

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


def robot_position_extractor():
    robot_position = RobotPosition()
    robot_position.x = odom.pose.pose.position.x
    robot_position.y = odom.pose.pose.position.y
    roll, pitch, yaw = euler_from_quaternion(odom.pose.pose.orientation)
    robot_position.theta = yaw


LEARNING_RATE = 0.0005 # learning rate
GAMMA = 0.99 # discount factor
REPLAY_BUFFER_SIZE = 200000 # replay buffer size
BATCH_SIZE = 1024 # minibatch size
IMAGE_SIZE = [60, 60]
INITIAL_EXPLORATION = 1.0
EPISODE_LENGTH = 300
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10 # the number to decide updating step
EPS_START = 0.9 # ?
EPS_END = 0.05 # ?
EPS_DECAY = 300 # ?

n_actions = 28

policy_net = DQN(screen_height, screen_width, n_actions).to(device) # for learning
target_net = DQN(screen_height, screen_width, n_actions).to(device) # for evaluating
target_net.load_state_dict(policy_net.state_dict()) # loading DQN model
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


def optimization_model():


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
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def reset():


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



def main():
    rospy.init_node('train', anonymous=True)

    ros = ROSNode()

    while not rospy.is_shutdown():
        rospy.spin()



if __name__ == '__main__':
    main()
