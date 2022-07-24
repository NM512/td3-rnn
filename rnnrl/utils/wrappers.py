from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np
from collections import deque

class PartialObservation(ObservationWrapper):
    def __init__(self, env, interval=3):
        super().__init__(env)
        self.count = 0
        self.interval = interval

    def observation(self, observation):
        self.count += 1
        if not self.count % self.interval == 0:
            observation = np.zeros_like(observation)
        return observation

class StackedObservation(ObservationWrapper):
    def __init__(self, env, stack_num=10):
        super().__init__(env)
        self.count = 0
        self.env = env
        self.stack_num = stack_num
        self._stacks = deque([], maxlen=stack_num)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_shape[0]*stack_num,), dtype=np.float32
            )
    
    def observation(self, observation):
        self._stacks.append(observation)
        return np.concatenate(list(self._stacks), axis=0)

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.stack_num):
            self._stacks.append(observation)
        return np.concatenate(list(self._stacks), axis=0)