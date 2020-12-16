


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
    def __init__(self, std=1):
        self.std = std
    
    def sample(self):
        return np.random.normal(scale=self.std)

class StochActionWrapper(ActionWrapper):
    def __init__(self, env, distrib='Gaussian', param=0.1):

        super(StochActionWrapper, self).__init__(env)
        if distrib == 'Gaussian':
            self.stoch_perturbation = Gaussian(std=param)


    def action(self, action):
        action = action + self.stoch_perturbation.sample()
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