import numpy as np
import random as rnd
from gym import Wrapper, spaces, ActionWrapper
import copy
from torch.distributions.normal import Normal


# class StochActionWrapper(Wrapper):
#     def __init__(self, env, ):
#         super(StochWrapper, self).__init__(env)

#         if isinstance(self.action_space, spaces.Discrete):
#             raise NotImplementedError
#         else:
#             high = np.tile(self.action_space.high, self.delay.max) 
#             low = np.tile(self.action_space.high, self.delay.max) 
#             shape = [self.delay.max*i for i in self.action_space.shape]
#             dtype = self.action_space.dtype
#             stored_actions = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


#     def reset(self, **kwargs):
#         return None

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)

#         # Sample new delay
#         _, n_obs = self.delay.sample()
#         # Get current state
#         self._hidden_obs.append(obs)

#         # Update extended state, rewards and hidden variables
#         self.extended_obs.append(action)
#         hidden_output = None
#         if n_obs > 0:
#             self.extended_obs[0] = self._hidden_obs[n_obs]
#             del self.extended_obs[1:(1+n_obs)]
#             hidden_output = np.array(self._hidden_obs[1:(1+n_obs)])
#             del self._hidden_obs[:n_obs]
        
#         self._reward_stock = np.append(self._reward_stock, reward)
#         if done:
#             reward_output = self._reward_stock
#             # reward_output = self._reward_stock # -> in this case, the sum is to be done in the algorithm
#         else:
#             reward_output = self._reward_stock[:n_obs]
#         self._reward_stock = np.delete(self._reward_stock, range(n_obs))

#         # Shaping the output
#         output = (self.extended_obs[0], np.array(self.extended_obs[1:], dtype=object))

#         return output, reward_output, done, (n_obs, hidden_output)

class Gaussian:
    def __init__(self, std=1.0):
        self.std = std
    
    def sample(self, action):
        return action + np.random.normal(scale=self.std)


class Uniform:
    def __init__(self, env_action_space, epsilon=0.1):
        self.epsilon = epsilon
        self.env_action_space = env_action_space
    
    def sample(self, action):
        if rnd.random() < self.epsilon:
            return self.env_action_space.sample()

        return action


class StochActionWrapper(ActionWrapper):
    def __init__(self, env, distrib='Gaussian', param=0.1):
        super(StochActionWrapper, self).__init__(env)
        if distrib == 'Gaussian':
            self.stoch_perturbation = Gaussian(std=param)
        elif distrib == 'Uniform':
            self.stoch_perturbation = Uniform(epsilon=param, env_action_space=self.env.action_space)

    def action(self, action):
        action = self.stoch_perturbation.sample(action)
        return action


class RandomActionWrapper(ActionWrapper):

    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):

        if rnd.random() < self.epsilon:

            print("Random!")

            return self.env.action_space.sample()

        return action