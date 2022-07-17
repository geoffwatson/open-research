from collections import deque
import random
from typing import Dict

import torch


class Memory:

    def __init__(self, max_size: int = 100000, use_cuda: bool = False) -> None:
        self.memory = deque(maxlen=max_size)
        self.use_cuda = use_cuda

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self, batch_size):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


class MemoryRecaller:
    """
    MemoryRecaller is used to fatch a batch of inputs for the PolicyUpdater.
    As the batch size, and items can depend on what needs to be updated, this is an interface
    that takes in the memory and uses a recall function to return the tensors required in a Dict.
    """

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def recall(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class BatchTransitionMemoryRecaller(MemoryRecaller):
    """
    Standard memory recall getting a random batch of unrelated transitions.
    """
    def __init__(self, memory: Memory, batch_size: int = 32):
        super(BatchTransitionMemoryRecaller, self).__init__(memory)
        self.batch_size = batch_size

    def recall(self) -> Dict[str, torch.Tensor]:
        state, next_state, action, reward, done = self.memory.recall(batch_size=self.batch_size)
        return {
            'state': state,
            'next_state': next_state,
            'action': action,
            'reward': reward,
            'done': done,
        }
