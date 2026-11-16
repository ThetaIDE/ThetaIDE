"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding



# ---------------------------------------------------------------------------------------------------
# MountainCar_v2
# - Description: Adapted from the MountainCar-v0
class MountainCar_v3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    spec = {'id': 'MountainCar_v3'}

    def __init__(self, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025
        self.viewer = None

        self.action_space = spaces.Discrete(3)

        # self.low = np.array([self.min_position, -self.max_speed, self.min_position, -self.max_speed], dtype=np.float32)
        # self.high = np.array([self.max_position, self.max_speed, self.max_position, self.max_speed], dtype=np.float32)

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.seed()

    def observation_space(self):
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# ---------------------------------------------------------------------------------------------------
# MountainCar_v2
# - Description: Adapted from the MountainCar-v0
class MountainCar_v2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    spec = {'id': 'MountainCar_v2'}

    def __init__(self, reform_n, reform_mode, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.reform_n = reform_n
        self.reform_mode = reform_mode
        self.viewer = None

        self.action_space = spaces.Discrete(3)

        self.resetLowHighValues()
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.seed()

    # Reset the low and high values for reformulated states
    def resetLowHighValues(self):
        tmp_low = [self.min_position, -self.max_speed]
        tmp_high = [self.max_position, self.max_speed]

        if self.reform_mode == 'original' or self.reform_mode == 'none':
            self.low = np.array(tmp_low, dtype=np.float32)
            self.high = np.array(tmp_high, dtype=np.float32)
        elif self.reform_mode == 'consecutive':
            self.low = np.array(tmp_low*self.reform_n, dtype=np.float32)
            self.high = np.array(tmp_high*self.reform_n, dtype=np.float32)
        elif self.reform_mode == 'skipping':
            self.low = np.array(tmp_low*2, dtype=np.float32)
            self.high = np.array(tmp_high*2, dtype=np.float32)


    def observation_space(self):
        return self.observation_space


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# ---------------------------------------------------------------------------------------------------
# GridWorld
class GridWorld():
    spec = {'id': 'GridWorld'}

    def __init__(self, action_num=4, grid_size=5):
        self.action_space = spaces.Discrete(action_num)
        self.low = np.array(np.ones([2]) * (grid_size-1), dtype=np.float32)
        self.high = np.array(np.zeros([2]), dtype=np.float32)

        # self.low = np.array(np.ones([grid_size*grid_size]), dtype=np.float32)
        # self.high = np.array(np.zeros([grid_size*grid_size]), dtype=np.float32)

        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.viewer = None
        self.seed()

    def observation_space(self):
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# ---------------------------------------------------------------------------------------------------
# Highway
class Highway():
    spec = {'id': 'Highway'}

    def __init__(self, action_num=5, dim=4):
        self.action_space = spaces.Discrete(action_num)
        self.low = np.array(np.ones([dim]), dtype=np.float32)
        self.high = np.array(np.zeros([dim]), dtype=np.float32)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.viewer = None
        self.seed()

    def observation_space(self):
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# ---------------------------------------------------------------------------------------------------
# CCHS
class CCHS():
    spec = {'id': 'CCHS'}

    def __init__(self, action_num=2, dim=42):
        self.action_space = spaces.Discrete(action_num)
        self.low = np.full((dim), np.inf) #np.array(np.inf([dim]), dtype=np.float32)
        self.high = np.full((dim), np.inf) #np.array(-np.inf([dim]), dtype=np.float32)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.viewer = None
        self.seed()

    def observation_space(self):
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# Mayo
class Mayo():
    spec = {'id': 'Mayo'}

    def __init__(self, action_num=2, dim=42):
        self.action_space = spaces.Discrete(action_num)
        self.low = np.full((dim), np.inf) #np.array(np.inf([dim]), dtype=np.float32)
        self.high = np.full((dim), np.inf) #np.array(-np.inf([dim]), dtype=np.float32)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.viewer = None
        self.seed()

    def observation_space(self):
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# MIMIC
class MIMIC():
    spec = {'id': 'MIMIC'}

    def __init__(self, action_num=2, dim=42):
        self.action_space = spaces.Discrete(action_num)
        self.low = np.full((dim), np.inf) #np.array(np.inf([dim]), dtype=np.float32)
        self.high = np.full((dim), np.inf) #np.array(-np.inf([dim]), dtype=np.float32)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.viewer = None
        self.seed()

    def observation_space(self):
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# MIMIC
class MIMICIII():
    spec = {'id': 'MIMICIII'}

    def __init__(self, action_num=2, dim=47):
        self.action_space = spaces.Discrete(action_num)
        self.low = np.full((dim), np.inf) #np.array(np.inf([dim]), dtype=np.float32)
        self.high = np.full((dim), np.inf) #np.array(-np.inf([dim]), dtype=np.float32)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.viewer = None
        self.seed()

    def observation_space(self):
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CusMLIRLEnv():
    # - demo_star： Input demonstrations --
    #               Format: [[(s01, a01), (s02, a02), ..., (a05, None)], [(s11, a11), (s12, a12), ..., (a15, None)], ...]
    # - _states: An array of M discrete states -- array([ 0,  1,  2,  3, ..., M])
    # - _actions: An array of N discrete actions -- array([0, 1, 2, 3])
    # - _p0s: An array of M values to indicate the state initial probabilities
    # - _t_mat: An array of M-N-M values to indicate the state-action-state transition probabilities
    # - _terminals: An array of M binary values to indicate whether each state is terminal state
    # - _gamma: discount factor

    def __init__(self, states, actions, p0s, transitions, terminals, gamma=0.99,):
        self._states = states
        self._actions = actions
        self._p0s = p0s
        self._terminal_state_mask = terminals
        self._t_mat = transitions
        self._gamma = gamma
