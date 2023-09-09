from lib.memory import Memory
from lib.memory import MemoryPostProcessor

import random
import gym


def test_memory_post_processor() -> None:
    env = gym.make('CartPole-v1', new_step_api=True)
    memory = Memory()

    for i in range(100):
        state = env.reset()
        done = False
        record_state = True
        first = True
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if random.random() > 0.5 or first:
                memory.record(state, action, reward, next_state)
                first = False
            else:
                memory.record(None, action, reward, next_state)
            record_state = not record_state
            state = next_state
        memory.terminate_episode()

    memory_post_processor = MemoryPostProcessor(discount_rate=0.98)
    sample_episodes = memory.sample(1)
    results = memory_post_processor.process(sample_episodes)

    for episode in results:
        for transition in episode.transitions:
            print(f'Reward: {transition.reward}')


test_memory_post_processor()