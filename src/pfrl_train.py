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

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

import cv2
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import kornia

import gym_ffmp
from gym_ffmp.envs.robot.config import RobotPose, RobotVelocity, RobotState, RobotAction
from gym_ffmp.envs.ffmp import FFMP

import time
import csv

import pfrl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### PARAM #####
# for ALL
maps = [None]

# for ROS
IMAGE_SIZE = 100
MAP_GRID_SIZE = 0.05
MAP_RANGE = 5.0
ROBOT_RSIZE = 0.13
SLEEP_TIME = 10

# for DDQN
ENV = 'FFMP-v0'
GAMMA = 0.95 # discount factor
MAX_STEPS = 100
NUM_EPISODES = 100000
LOG_DIR = "./logs/train06"
BATCH_SIZE = 1024 # minibatch size
# BATCH_SIZE = 512# minibatch size
CAPACITY = 200000 # replay buffer size
# INPUT_CHANNELS = 12 #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)
# INPUT_CHANNELS = 3 #[channel] = (occupancy(MONO) + flow(xy)) * series(1 steps)
# INPUT_CHANNELS = 1 # only temporal_bev_image
INPUT_CHANNELS = 2 # two steps of temporal_bev_image
NUM_ACTIONS = 28
LEARNING_RATE = 0.0005 # learning rate
# LOSS_THRESHOLD = 0.1 # threshold of loss
# LOSS_THRESHOLD = 0.000000001 # threshold of loss
LOSS_MEMORY_CAPACITY = 10
REACH_RATE_THRESHOLD = 0.80
REACH_MEMORY_CAPACITY = 10
UPDATE_TARGET_EPISODE = 2
MAX_TOTAL_STEP = 100000
MODEL_PATH = './model/train06/model.pth'
################

class ROSNode():
    def __init__(self):
        self.sub_temporal_bev = rospy.Subscriber("/bev/temporal_bev_image", Image, self.temporal_bev_image_callback)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.sub_start_goal = rospy.Subscriber("/start_goal_points", PoseArray, self.pose_array_callback)
        self.sub_laser = rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        self.sub_start_flag = rospy.Subscriber("/is_start_episode", Bool, self.is_start_callback)
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pub_done_flag = rospy.Publisher("/episode_watcher/is_finish_episode", Bool, queue_size=1)

        self.odom = Odometry()
        self.global_start = Pose()
        self.global_goal = Pose()
        self.bridge = CvBridge()
        self.done_flag = Bool()
        self.scan_data = [None]
        self.pre_robot_pose = RobotPose(0.0, 0.0, 0.0)
        self.map_grid_size = MAP_GRID_SIZE
        self.map_range = MAP_RANGE
        self.robot_rsize = ROBOT_RSIZE
        self.temporal_bev_map = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
        self.current_odom_time = 0.0
        self.previous_odom_time = 0.0
        self.odom_dt = 0.0

        self.temporal_bev_image_callback_flag = False
        self.odom_callback_flag = False
        self.posearray_callback_flag = False
        self.laser_callback_flag = False
        self.is_start_callback_flag = True

    def is_start_callback(self, msg):
        self.is_start_callback_flag = True

    def temporal_bev_image_callback(self, msg):
        # print("temporal_bev_image_callback")
        cv_temporal_bev_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        tensor_bev_image = kornia.image_to_tensor(cv_temporal_bev_image, keepdim=True).float()
        self.temporal_bev_map = tensor_bev_image
        self.temporal_bev_image_callback_flag = True

    def odom_callback(self, msg):
        # print("odom_callback")
        self.odom = msg
        self.odom_callback_flag = True

    def pose_array_callback(self, msg):
        # print("pose_array_callback")
        self.global_start = msg.poses[0]
        self.global_goal = msg.poses[1]
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

    def laser_callback(self, msg):
        for i in range(len(msg.ranges)):
            if msg.ranges[i] != float('inf') and msg.ranges[i]:
                self.scan_data.append(msg.ranges[i])
                # print("ROS:scan_data[0] =" ,self.scan_data[0])
                self.laser_callback_flag = True

    def done_flag_publisher(self, is_done):
        print("ros.done_flag =", is_done)
        self.done_flag = is_done
        self.pub_done_flag.publish(self.done_flag)

    def robot_position_extractor(self):
        self.current_odom_time = float(self.odom.header.stamp.to_sec())
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

    def robot_displacement_calculator(self, robot_pose, is_first):
        if is_first:
            self.pre_robot_pose = copy.deepcopy(robot_pose)
        linear_displacement = math.sqrt(math.pow(robot_pose.x - self.pre_robot_pose.x, 2) + math.pow(robot_pose.y - self.pre_robot_pose.y, 2))
        angular_displacement = self.pi_to_pi(robot_pose.yaw - self.pre_robot_pose.yaw)
        self.pre_robot_pose = copy.deepcopy(robot_pose)
        return np.array([linear_displacement, angular_displacement])
    
    def gazebo_pause_client(self):
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            srv_gazebo_pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
            res = srv_gazebo_pause()
            print("res", res)
            print("pause gazebo!")
        except rospy.ServiceException:
            print("Service call failed:")

    def gazebo_unpause_client(self):
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            srv_gazebo_unpause = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
            res = srv_gazebo_unpause()
            print("res", res)
            print("unpause gazebo!")
        except rospy.ServiceException:
            print("Service call failed:")





class Network(nn.Module):
    def __init__(self, input_channels, outputs):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=8)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=8)
        self.fc1 = nn.Linear(5, 67)
        self.fc2 = nn.Linear(6400, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_ea = nn.Linear(512, outputs) # Elements A(s,a) that depend on your actions.
        self.fc4_ev = nn.Linear(512, 1) # Elements V(s) that are determined only by the state.

    def forward(self, state):
        state_m = state[:-5]
        state_m = torch.reshape(state_m, (2, IMAGE_SIZE, IMAGE_SIZE))
        state_g = state[-4:-3]
        state_v = state[-2:]
        # print("state_m.size() = ", state_m.size())
        # print("state_g.size() = ", state_g.size())
        # print("state_v.size() = ", state_v.size())
        x_m = F.relu(self.conv1(state_m))
        # print("conv1 -> x_m.size() = ", x_m.size())
        x_m = F.relu(self.conv2(x_m))
        # print("conv2 -> x_m.size() = ", x_m.size())
        x_m = F.relu(self.conv3(x_m))
        # print("conv3 -> x_m.size() = ", x_m.size())

        x_gv_ = torch.cat((state_g, state_v), 1)
        x_gv_ = F.relu(self.fc1(x_gv_))
        # print("x_gvt_.size() = ", x_gvt_.size())
        convw = x_m.shape[2]
        convh = x_m.shape[3]
        # print("convw =", convw, ", convh =", convh)
        x_gv = torch.empty(BATCH_SIZE, convw, convh)
        
        grid_num = convw
        for i in range(grid_num):
            value = x_gvt_[0][i].item()
            tile_gvt = torch.full((convw, convh), value)
            x_gvt = torch.stack(tuple(tile_gvt), 0)
        
        # print("x_gv.size() = ", x_gv.size())

        x_m = x_m.to('cpu')
        x_gvt = x_gvt.to('cpu')
        x_pls = x_m + x_gvt
        x_pls = x_pls.to(device)

        x_pls = F.relu(self.conv4(x_pls))
        x_pls = F.relu(self.conv4(x_pls))
        x_pls = F.relu(self.conv4(x_pls))
        # print("x_pls.size() = ", x_pls.size())

        ## batch_flattened_length = x_pls.shape[1] * x_pls.shape[2] * x_pls.shape[3]
        ## x = torch.empty(BATCH_SIZE, batch_flattened_length)
        x = torch.flatten(x_pls, start_dim=1)
        # print("x.size() = ", x.size())

        x = F.relu(self.fc2(x))
        # print("fc2 -> x.size() = ", x.size())
        x = F.relu(self.fc3(x))
        # print("fc3 -> x.size() = ", x.size())
        adv = self.fc4_ea(x)
        val = self.fc4_ev(x)
        ## adv = torch.unsqueeze(adv, 0)
        # print("adv.size() = ", adv.size())
        ## val = torch.unsqueeze(val, 0)
        # print("val.size() = ", val.size())
        adv = adv.to('cpu')
        val = val.to('cpu')

        output = adv + val - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        output = output.to(device)
        # print("output.size()", output.size())

        return output



class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.agent = Agent()
        self.map_memory = []
        self.loss_memory = []
        self.loss_memory_capacity = LOSS_MEMORY_CAPACITY
        self.loss_ave = None
        # self.loss_convergence = False
        self.reach_rate = 0.0

    def check_convergence(self):
        if len(self.loss_memory) > self.loss_memory_capacity:
            del self.loss_memory[0]
        
        self.loss_memory.append(self.agent.brain.loss)
        self.loss_ave = sum(self.loss_memory) / len(self.loss_memory)

        # if self.loss_ave < LOSS_THRESHOLD:
        #     self.loss_convergence = True
    def make_temporal_maps(self, import_map, is_first):
        if is_first:
            self.map_memory.clear()
            for i in range(INPUT_CHANNELS):
                self.map_memory.append(import_map)
        else:
            self.map_memory.append(import_map)
            del self.map_memory[0]

        temporal_maps = torch.cat(self.map_memory, 0)
        ## temporal_maps.size() = torch.Size([2, H, W])
        print("temporal_maps.size() = ", temporal_maps.size())
        return temporal_maps

class UseTensorBord:
    def __init__(self, path):
        self.writer = SummaryWriter(log_dir=path)
        # self.log_loss = []


def main():
    rospy.init_node('train', anonymous=True)
    ros = ROSNode()
    r = rospy.Rate(100)

    train_env = Environment()
    
    # q_func = Network(INPUT_CHANNELS, NUM_ACTIONS).to(device)
    q_func = Network(INPUT_CHANNELS, NUM_ACTIONS)

    agent = pfrl.agents.DoubleDQN(\
            q_func,\
            optimizer=torch.optim.Adam(q_func.parameters(), eps=1e-2),\
            replay_buffer=pfrl.replay_buffers.ReplayBuffer(CAPACITY),\
            gamma = GAMMA,\
            explorer=pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample),\
            replay_start_size = BATCH_SIZE,\
            update_interval=1,\
            target_update_interval=100,\
            gpu = device\
            )

    episode = 0
    step = 0
    total_step = 0
    total_reward = 0
    reach_times = np.empty(0)
    is_first = True
    is_done = True
    action = None
    action_id = 3
    state = None
    is_complete = False

    tensor_board = UseTensorBord(LOG_DIR)
    
    f = open(LOG_DIR + '/log_loss.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['episode', 'total_step', 'loss_value', 'train_env.reach_rate', 'numpy_reward', 'relative_goal_distance', 'relative_goal_direction', 'odom_x', 'odom_y', 'odom_yaw'])

    print("ros : start!")
    ros.gazebo_unpause_client()
    while not rospy.is_shutdown():
        # print("[0] ros.temporal_bev_image_callback_flag:", ros.temporal_bev_image_callback_flag)
        # print("[1] ros.odom_callback_flag:", ros.odom_callback_flag)
        # print("[2] ros.posearray_callback_flag:", ros.posearray_callback_flag)
        # print("[3] ros.laser_callback_flag:", ros.laser_callback_flag)
        if ros.temporal_bev_image_callback_flag  and ros.odom_callback_flag and ros.posearray_callback_flag and ros.laser_callback_flag and ros.is_start_callback_flag:
            print("ros flags : OK")

            ros.gazebo_pause_client()
            if is_first:
                ros.previous_odom_time = ros.current_odom_time

            robot_pose = ros.robot_position_extractor() # numpy
            relative_goal = ros.relative_goal_calculator(robot_pose) # numpy
            tmp_relative_goal = copy.deepcopy(relative_goal) # numpy
            displacement = ros.robot_displacement_calculator(robot_pose, is_first) # numpy
            odom_dt = ros.current_odom_time - ros.previous_odom_time
            # ros.odom_dt = np.array([ros.current_odom_time - ros.previous_odom_time])
            velocity = np.array([displacement[0] / odom_dt, displacement[1] / odom_dt])
            flow_map = ros.temporal_bev_map.clone() #tensor

            # observe_m = flow_map
            observe_m = train_env.make_temporal_maps(flow_map, is_first)
            observe_m = torch.unsqueeze(observe_m, 0)
            observe_m = observe_m.to(device)
            observe_m = torch.reshape(observe_m, (-1,))
            # print("observe_m.size() = ", observe_m.size())
            observe_g = torch.from_numpy(tmp_relative_goal).type(torch.FloatTensor)
            observe_g = torch.unsqueeze(observe_g, 0)
            observe_g = observe_g.to(device)
            # print("observe_g.size() = ", observe_g.size())
            observe_v = torch.from_numpy(velocity).type(torch.FloatTensor)
            observe_v = torch.unsqueeze(observe_v, 0)
            observe_v = observe_v.to(device)
            # print("observe_t.size() = ", observe_t.size())
            # observe_t = torch.from_numpy(ros.odom_dt).type(torch.FloatTensor)
            # observe_t = torch.unsqueeze(observe_t, 0)
            # observe_t = observe_t.to(device)
            observe = torch.cat((observe_m, observe_g), 0)
            observe = torch.cat((observe, observe_v), 0)

            if is_first:
                reward = 0
                action_id = 3
                state = observe
            
            state = state.to(device)

            action = agent.act(state)
            action_id = action.item()

            print("relative_goal =", relative_goal, "[m]")

            numpy_reward, is_done, is_goal = train_env.env.rewarder2(ros.scan_data, relative_goal, is_first)

            if is_goal:
                reach_times = np.append(reach_times, 1.0)
            else:
                reach_times = np.append(reach_times, 0.0)

            if len(reach_times) > REACH_MEMORY_CAPACITY:
                reach_times = np.delete(reach_times, 0, None)
            
            train_env.reach_rate = np.average(reach_times)
            
            # reward = torch.as_tensor(numpy_reward.astype('float32')).clone()
            reward = torch.as_tensor(float(numpy_reward)).clone()
            reward = torch.unsqueeze(reward, 0)

            reset = step == MAX_STEPS
            agent.observe(observe, reward, is_done, reset)

            is_first = False

            if step == MAX_STEPS:
                is_done = True

            ros.temporal_bev_image_callback_flag = False
            ros.odom_callback_flag = False
            ros.posearray_callback_flag = False
            ros.laser_callback_flag = False
            ros.scan_data.clear()

            ros.previous_odom_time = ros.current_odom_time
            
            if is_done:
                episode += 1
                step = 0
                
                if train_env.agent.brain.loss != None:
                    print("is_done : ", is_done)
                    loss_value = train_env.agent.brain.loss.item()
                    print("EPISODE :", episode)
                    print("loss =", loss_value)
                    print("reach_rate =", train_env.reach_rate)

                    # tensor_board.log_loss.append(loss_value)
                    # tensor_board.writer.add_scalar('ours', tensor_board.log_loss[episode], episode)
                    tensor_board.writer.add_scalar('LOSS [Flow Field Based Motion Planner]', loss_value, episode)
                    tensor_board.writer.add_scalar('REACH_RATE [Flow Field Based Motion Planner]', train_env.reach_rate, episode)
                    tensor_board.writer.add_scalar('REWARD [Flow Field Based Motion Planner]', total_reward, episode)
                    tensor_board.writer.add_scalar('RELATIVE_GOAL_DISTANCE [Flow Field Based Motion Planner]', relative_goal[0], episode)
                    writer.writerow([episode, total_step, loss_value, train_env.reach_rate, total_reward, relative_goal[0], relative_goal[1], ros.robot_pose.x, ros.robot_pose.y)

                    state = None

                    if train_env.reach_rate > REACH_RATE_THRESHOLD:
                        torch.save(train_env.agent.brain.main_q_network.state_dict(), MODEL_PATH)
                        print("COMPLETED TO LEARN! (reach_rate)")
                        print("SAVED MODEL!")
                        is_complete = True
                    if episode % 10 == 0:
                        torch.save(train_env.agent.brain.main_q_network.state_dict(), MODEL_PATH)
                        print("COMPLETED TO LEARN! (reach_rate)")
                        print("SAVED MODEL!")


                ros.gazebo_unpause_client()
                ros.cmd_vel_publisher(0.0, 0.0)
                ros.done_flag_publisher(is_done)
                total_reward = 0
                is_first = True
                is_done = False
                ros.is_start_callback_flag = False
            else:
                print("episode:", episode, ", step :", step)
                print("total_step:",total_step , ", loss:",train_env.agent.brain.loss)
                linear_v = train_env.agent.action.commander(action_id).linear_v
                angular_v = train_env.agent.action.commander(action_id).angular_v
                ros.gazebo_unpause_client()
                print("linear_v =", linear_v, "[m/s]")
                print("angular_v =", angular_v, "[m/s]")
                ros.cmd_vel_publisher(linear_v, angular_v)
                ros.done_flag_publisher(is_done)

                state = observe

                step += 1
                total_step += 1
                total_reward += numpy_reward

        if is_complete:
            print("complete")
            tensor_board.writer.close()
            break

        if total_step > MAX_TOTAL_STEP:
            print("Game Over..................")
            break

        r.sleep()

    print("Try to save model!")
    torch.save(train_env.agent.brain.main_q_network.state_dict(), MODEL_PATH)
    print("Saved model!")
    tensor_board.writer.close()
    f.close()

if __name__ == '__main__':
    main()
