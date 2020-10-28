#coding:utf-8

import numpy as np
import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from robot import RobotPosition, RobotVelocity, RobotState, RobotAction


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
        # [2-2] relative_goal [distance, orientation]
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
        #                                self.position_min.yaw])
        # self.position_high = np.array([self.position_max.x, \
        #                                self.position_max.y, \
        #                                self.position_max.yaw])
        # self.position_space = spaces.Dict
 
    def reset(self, relative_goal_info):
        self.action = RobotAction()
        self.state = np.array([[0.0, 0.0], relative_goal_info, self.action.cmd[3], False, False])



    def is_collision(self, robot_grids, local_map_info):
        is_collision = False
        for itr in robot_grids:
            i = itr[0]
            j = itr[1]
            if local_map_info[i][j][0] > 0:
                is_collision = True
                break
        return is_collision
               

    def is_goal(self, cur_relative_goal_dist):
        dist_threshold = 0.2 #[m]
        is_goal = False

        if cur_relative_goal_dist < dist_threshold:
            is_goal = True

        return is_goal

       
    def reward(self, relative_goal_info, is_collision, is_goal, is_first):
        r_g = 0
        r_c = 0
        r_t = 0
        r_arr = 500
        r_col = -500
        r_s = -5
        epsilon = 10

        global pre_relative_goal_dist
        if is_first == True:
            pre_relative_goal_dist = relative_goal_info[0]
        
        cur_relative_goal_dist = relative_goal_info[0] #[0]:range, [1]:orientation

        if is_goal:
            r_g = r_arr
        else:
            r_g = epsilon * (pre_relative_goal_dist - cur_relative_goal_dist)

        if is_collision == True:
            r_c = r_coll
        else:
            r_c = 0

        r_t = r_g + r_c + r_s

        return r_t


    def is_done(self, is_collision, is_goal):
        if is_collision or is_goal:
            return True
        else
            return False


    def step(self, local_map_info, relative_goal_info, velocity_info, is_first, action):
        self.state[0] = local_map_info
        self.state[1] = relative_goal_info
        self.state[2] = velocity_info
        self.state[3] = is_collision(local_map_info)
        self.state[4] = is_goal(relative_goal_info[0]) #[0]:distance, [1]:orientation
        self.observation = np.array([self.state[0], self.state[1], action])
        self.reward = self.reward(self.state[1], self.state[3], self.state[4], is_first)
        self.is_done = is_done(self.state[3], self.state[4])

        return self.observation, self.reward, self.is_done









