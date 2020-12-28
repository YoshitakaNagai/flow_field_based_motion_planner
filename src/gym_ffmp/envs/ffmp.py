#coding:utf-8

import numpy as np
import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# import sys
# import os
# import robot
from .robot.config import RobotPose, RobotVelocity, RobotState, RobotAction

MAP_RANGE = 5.0 # [m]
MAP_GRID_NUM = 100 # [grids]
MAP_CHANNELS = 1 #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)
ROBOT_RSIZE = 0.13 # [m]
MAP_RESOLUTION = 0.05
GOAL_THRESHOLH = 0.5


class FFMP(gym.Env):
    # metadata = {'render.modes' : ['human', 'rgb_array']}

    def __init__(self):
        # super().__init__()

        # [1] action_space
        self.action = RobotAction()
        self.action_low  = np.array([self.action.cmd[0].linear_v, self.action.cmd[0].angular_v])
        self.action_high = np.array([self.action.cmd[27].linear_v, self.action.cmd[27].angular_v])
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # [2] observation_space
        # [2-1] local_map
        self.map_range = MAP_RANGE #[m]
        self.map_grid_num = MAP_GRID_NUM #[grids]
        self.map_grid_size = MAP_RESOLUTION
        self.map_channels = MAP_CHANNELS #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)
        self.map_low  = np.full((self.map_grid_num, self.map_grid_num, self.map_channels), 0)
        self.map_high = np.full((self.map_grid_num, self.map_grid_num, self.map_channels), 255)
        # [2-2] relative_goal [distance, orientation]
        self.goal_low  = np.array([0.0, 0.0])
        self.goal_high = np.array([math.sqrt(2.0) * self.map_range, math.pi])
        # [2-3] velocity
        self.velocity_low  = self.action_low
        self.velocity_high = self.action_high
        # [2-4] colision
        self.robot_rsize = ROBOT_RSIZE #[m]
        # self.robot_grids = np.empty([0, 0], dtype=int32)
        self.collision_low  = False
        self.collision_high = True
        # observation_space
        self.observation = np.array([self.goal_high, self.action_low])
        self.observation_space = spaces.Dict({
            "local_map": spaces.Box(self.map_low, self.map_high, dtype=np.int32), \
            "relative_goal": spaces.Box(self.goal_low, self.goal_high, dtype=np.float32), \
            "velocity": spaces.Box(self.velocity_low, self.velocity_high, dtype=np.float32)})

        # [3] state_space
        self.state_space = spaces.Dict({
            "local_map": spaces.Box(self.map_low, self.map_high, dtype=np.int32), \
            "relative_goal": spaces.Box(self.goal_low, self.goal_high, dtype=np.float32), \
            "velocity": spaces.Box(self.velocity_low, self.velocity_high, dtype=np.float32)})
 
        # [4] position_space (sub)
        # self.position_min = RobotPose(0.0, 0.0, math.radians(0))
        # self.position_max = RobotPose(self.map_range, self.map_range, math.radians(360))
        # self.position_low  = np.array([self.position_min.x, \
        #                                self.position_min.y, \
        #                                self.position_min.yaw])
        # self.position_high = np.array([self.position_max.x, \
        #                                self.position_max.y, \
        #                                self.position_max.yaw])
        # self.position_space = spaces.Dict
 
    # def reset(self, local_map_info, relative_goal_info):
    # def reset(self):
    #     self.action = self.action_low
    #     # self.observation = np.array([relative_goal_info, self.action])
    #     reward = 0
    #     
    #     return reward

    def is_collision(self, local_map_info):
        self.robot_grids = []
        for i in range(self.map_grid_num):
            for j in range(self.map_grid_num):
                x_dist_pow = math.pow(i * self.map_grid_size - 0.5 * self.map_range, 2)
                y_dist_pow = math.pow(j * self.map_grid_size - 0.5 * self.map_range, 2)
                dist = math.sqrt(x_dist_pow + y_dist_pow)
                #print("dist[",i,"][",j,"] = ", dist)
                if dist <= self.robot_rsize:
                    self.robot_grids.append(np.array([i, j]))
                    #print("INSIDE ROBOT ... dist = ", dist)
        is_collide = False
        for itr in range(len(self.robot_grids)):
            #print(self.robot_grids[itr])
            i = self.robot_grids[itr][0]
            j = self.robot_grids[itr][1]
            if local_map_info[i,j] > 0:
                is_collide = True
                break
        
        return is_collide
               

    def is_collision2(self, scan_data):
        is_collide = False
        for i in range(len(scan_data)):
            if scan_data[i]:
                if scan_data[i] < ROBOT_RSIZE:
                    is_collide = True
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    break

        return is_collide


    def is_goal(self, cur_relative_goal_dist):
        dist_threshold = GOAL_THRESHOLH #[m]
        is_goal = False

        if cur_relative_goal_dist < dist_threshold:
            is_goal = True

        return is_goal

       
    def reward_calculator(self, relative_goal_info, is_collision, is_goal, is_first):
        r_g = 0
        r_c = 0
        r_t = 0
        r_arr = 1000
        r_col = -500
        r_s = -5
        epsilon = 50

        global pre_relative_goal_dist
        if is_first:
            pre_relative_goal_dist = relative_goal_info[0]
        
        cur_relative_goal_dist = relative_goal_info[0] #[0]:range, [1]:orientation

        if is_goal:
            r_g = r_arr
        else:
            r_g = epsilon * (pre_relative_goal_dist - cur_relative_goal_dist)

        if is_collision:
            r_c = r_col
        else:
            r_c = 0

        r_t = r_g + r_c + r_s

        return r_t


    def is_done(self, is_collision, is_goal):
        if is_collision or is_goal:
            return True
        else:
            return False


    def rewarder(self, local_map_info, relative_goal_info, is_first):
        is_collide = self.is_collision(local_map_info)
        #print("is_collide : ", is_collide)
        is_goal = self.is_goal(relative_goal_info[0]) #[0]:distance, [1]:orientation

        # self.observation = np.array([relative_goal_info, action])
        reward = self.reward_calculator(relative_goal_info, is_collide, is_goal, is_first)
        is_done = self.is_done(is_collide, is_goal)

        return reward, is_done


    def rewarder2(self, scan_data, relative_goal_info, is_first):
        is_collide = self.is_collision2(scan_data)
        # print("is_collide : ", is_collide)
        is_goal = self.is_goal(relative_goal_info[0]) #[0]:distance, [1]:orientation

        # self.observation = np.array([relative_goal_info, action])
        reward = self.reward_calculator(relative_goal_info, is_collide, is_goal, is_first)
        is_done = self.is_done(is_collide, is_goal)

        return reward, is_done, is_goal
