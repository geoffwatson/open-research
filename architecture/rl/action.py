import numpy as np
import torch


class ActionSelector:

    def select_action(self, state, network):
        raise NotImplementedError


class EpsilonGreedyActionSelector(ActionSelector):

    def __init__(self, action_dim, use_cuda, exploration_decay_rate, exploration_rate_min):
        self.action_dim = action_dim
        self.use_cuda = use_cuda

        self.exploration_decay_rate = exploration_decay_rate
        self.exploration_rate_min = exploration_rate_min

        self.exploration_rate = 1.0

    def select_action(self, state, network):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state.__array__()
            state = torch.tensor(state)
            if self.use_cuda:
                state = state.cuda()
            state = state.unsqueeze(0)
            action_values = network(state, model_type="online")

            action_idx = torch.argmax(action_values, dim=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        return action_idx