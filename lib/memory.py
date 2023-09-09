from typing import Optional

import numpy as np


class Transition:
    def __init__(
      self,
      state: np.ndarray,
      action: np.ndarray,
      reward: float,
      next_state: np.ndarray,
    ) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state


class Episode:
    def __init__(self) -> None:
        self.transitions = []

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def get_transitions(self) -> list[Transition]:
        return self.transitions


class Memory:
    def __init__(self, max_size: int = 1000000) -> None:
        self.episodes = []
        self.index = 0
        self.max_size = max_size
        self.current_episode = Episode()

    def record(
        self,
        state: Optional[np.ndarray],
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        self.current_episode.append(Transition(state, action, reward, next_state))

    def terminate_episode(self) -> None:
        self.episodes.append(self.current_episode)
        self.index = (self.index + 1) % self.max_size
        self.current_episode = Episode()

    def sample(self, batch_size: int = 1) -> list[Episode]:
        indexes = np.random.choice(self.index, batch_size)
        return [self.episodes[idx] for idx in indexes]


class MemoryPostProcessor:

    def __init__(self, discount_rate: float = 0.9) -> None:
        self.discount_rate = discount_rate

    def process(self, episodes: list[Episode]) -> list[Episode]:
        formatted = []
        for episode in episodes:
            current_episode = Episode()
            states, actions, reward, next_states = [], [], 0, []
            for transition in episode.get_transitions():
                state = transition.state
                if state is not None:
                    if len(states) > 0:
                        current_episode.append(Transition(
                            np.stack(states, axis=0),
                            np.stack(actions, axis=0),
                            reward,
                            np.stack(next_states, axis=0),
                        ))
                        states, actions, reward, next_states = [], [], 0, []
                    states.append(state)
                actions.append([transition.action])
                reward = reward + transition.reward
                next_states.append(transition.next_state)

            current_episode.append(Transition(
                np.stack(states, axis=0),
                np.stack(actions, axis=0),
                reward,
                np.stack(next_states, axis=0),
            ))

            reward = 0.0
            for transition in reversed(current_episode.transitions):
                transition.reward = transition.reward + reward * self.discount_rate
                reward = transition.reward

            formatted.append(current_episode)

        return formatted
