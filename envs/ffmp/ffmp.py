#coding:utf-8

import numpy as np
import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import os

from function.raycast import *


class RobotPosition(object):
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta


class RobotVelocity(object):
    def __init__(self, linear_v, angular_v):
        self.linear = linear_v
        self.angular = angular_v


class RobotState(RobotPosition, RobotVelocity):
    def __init__(self, x, y, theta, linear_v, angular_v):
        self.robot_position = RobotPosition(x, y, theta)
        self.robot_velocity = RobotVelocity(linear_v, angular_v)


class RobotAction(object):
    def __init__(self):
        self.cmd = [RobotVelocity()] * 28
        self.cmd[0].linear = 0.0
        self.cmd[0].angular = -0.9
        self.cmd[1].linear = 0.0
        self.cmd[1].angular = -0.6
        self.cmd[2].linear = 0.0
        self.cmd[2].angular = -0.3
        self.cmd[3].linear = 0.0
        self.cmd[3].angular = 0.0
        self.cmd[4].linear = 0.0
        self.cmd[4].angular = 0.3
        self.cmd[5].linear = 0.0
        self.cmd[5].angular = 0.6
        self.cmd[6].linear = 0.0
        self.cmd[6].angular = 0.9
        self.cmd[7].linear = 0.2
        self.cmd[7].angular = -0.9
        self.cmd[8].linear = 0.2
        self.cmd[8].angular = -0.6
        self.cmd[9].linear = 0.2
        self.cmd[9].angular = -0.3
        self.cmd[10].linear = 0.2
        self.cmd[10].angular = 0.0
        self.cmd[11].linear = 0.2
        self.cmd[11].angular = 0.3
        self.cmd[12].linear = 0.2
        self.cmd[12].angular = 0.6
        self.cmd[13].linear = 0.2
        self.cmd[13].angular = 0.9
        self.cmd[14].linear = 0.4
        self.cmd[14].angular = -0.9
        self.cmd[15].linear = 0.4
        self.cmd[15].angular = -0.6
        self.cmd[16].linear = 0.4
        self.cmd[16].angular = -0.3
        self.cmd[17].linear = 0.4
        self.cmd[17].angular = 0.0
        self.cmd[18].linear = 0.4
        self.cmd[18].angular = 0.3
        self.cmd[19].linear = 0.4
        self.cmd[19].angular = 0.6
        self.cmd[20].linear = 0.4
        self.cmd[20].angular = 0.9
        self.cmd[21].linear = 0.6
        self.cmd[21].angular = -0.9
        self.cmd[22].linear = 0.6
        self.cmd[22].angular = -0.6
        self.cmd[23].linear = 0.6
        self.cmd[23].angular = -0.3
        self.cmd[24].linear = 0.6
        self.cmd[24].angular = 0.0
        self.cmd[25].linear = 0.6
        self.cmd[25].angular = 0.3
        self.cmd[26].linear = 0.6
        self.cmd[26].angular = 0.6
        self.cmd[27].linear = 0.6
        self.cmd[27].angular = 0.9



class FFMP(gym.Env):
    # metadata = {'render.modes' : ['human', 'rgb_array']}

    def __init__(self):
        super().__init__()

        # [1] action_space
        self.action = RobotAction()
        self.action_low  = np.array([self.action.cmd[0].linear, self.action.cmd[0].angular])
        self.action_high = np.array([self.action.cmd[27].linear, self.action.cmd[27].angular])
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # [2] observation_space
        # [2-1] local_map
        self.map_range = 10.0 #[m]
        self.map_grid_num = 60 #[grid]
        self.map_grid_size = self.map_range / (float)self.map_grid_num
        self.map_channels = 12 #[channel] = (occupancy(MONO) + flow(RGB)) * series(3 steps)
        self.map_low  = np.full((self.map_grid_num, self.map_grid_num, self.map_channels), 0)
        self.map_high = np.full((self.map_grid_num, self.map_grid_num, self.map_channels), 255)
        # [2-2] relative_goal [range, theta]
        self.goal_low  = np.array([0.0, 0.0])
        self.goal_high = np.array([sqrt(2.0) * self.map_range, math.pi)
        # [2-3] velocity
        self.velocity_low  = self.action_low
        self.velocity_high = self.action_high
        # [2-4] colision
        self.robot_rsize = 0.2 #[m]
        self.robot_grids = np.empty([0, 0], dtype=int32)
        for i in range(self.map_grid_num):
            for j in range(self.map_grid_num):
                if math.sqrt(math.pow(i * self.map_grid_size - 0.5 * self.map_range, 2) \
                            + math.pow(j * self.map_grid_size - 0.5 * self.map_range, 2) \
                    <= self.robot_rsize:
                    self.robot_grids = np.append(self.robot_grids, [i, j])
        self.collision_low  = False
        self.collision_high = True
        # observation_space
        self.observation_space = spaces.Dict({
            "local_map": spaces.Box(self.map_low, self.map_high, dtype=np.int32), \
            "relative_goal": spaces.Box(self.goal_low, self.goal_high, dtype=np.float32), \
            "velocity": spaces.Box(self.velocity_low, self.velocity_high, dtype=np.float32)})

        # [3] state_space
        self.state_space = spaces.Dict({
            "local_map": spaces.Box(self.map_low, self.map_high, dtype=np.int32), \
            "relative_goal": spaces.Box(self.goal_low, self.goal_high, dtype=np.float32), \
            "velocity": spaces.Box(self.velocity_low, self.velocity_high, dtype=np.float32), \
            "collision": spaces.Box(self.collision_low, self.collision_high, dtype=bool)})
 
        # [4] position_space (sub)
        # self.position_min = RobotPosition(0.0, 0.0, math.radians(0))
        # self.position_max = RobotPosition(self.map_range, self.map_range, math.radians(360))
        # self.position_low  = np.array([self.position_min.x, \
        #                                self.position_min.y, \
        #                                self.position_min.theta])
        # self.position_high = np.array([self.position_max.x, \
        #                                self.position_max.y, \
        #                                self.position_max.theta])
        # self.position_space = spaces.Dict
 
    def reset(self, relative_goal_info):
        self.action = RobotAction()
        self.state = np.array([[0.0, 0.0], relative_goal_info, self.action.cmd[3], False])

    def step(self, local_map_info, relative_goal_info, velocity_info, is_first, action):
        self.state[0] = local_map_info
        self.state[1] = relative_goal_info
        self.state[2] = velocity_info
        self.state[3] = self.is_collision(local_map_info)
        self.observation = np.array([self.state[0], self.state[1], action])
        self.reward = self.reward(self.state[1], self.state[3], is_first)

        return self.observation, self.reward


    def is_collision(self, robot_grids, local_map_info):
        is_collision = False
        for itr in robot_grids:
            i = itr[0]
            j = itr[1]
            if local_map_info[i][j][0] > 0:
                is_collision = True
                break
        return is_collision
               
       
    def reward(self, relative_goal_info, is_collision, is_first):
        r_g = 0
        r_c = 0
        r_t = 0
        r_arr = 500
        r_col = -500
        r_s = -5
        dist_threshold = 0.2 #[m]
        epsilon = 10

        global pre_relative_goal_dist
        if is_first == True:
            pre_relative_goal_dist = relative_goal_info[0]
        
        cur_relative_goal_dist = relative_goal_info[0]

        if cur_relative_goal_dist < dist_threshold:
            r_g = r_arr
        else:
            r_g = epsilon * (pre_relative_goal_dist - cur_relative_goal_dist)

        if is_collision == True:
            r_c = r_coll
        else:
            r_c = 0

        r_t = r_g + r_c + r_s

        return r_t















