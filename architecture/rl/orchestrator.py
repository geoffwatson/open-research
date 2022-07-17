# !pip install gym-super-mario-bros==7.3.0
from pathlib import Path

import torch

import datetime

from architecture.rl.metrics import MetricLogger
from architecture.rl.agent import Agent
from architecture.rl.action import EpsilonGreedyActionSelector
from architecture.rl.env import SkipFrame
from architecture.rl.env import GrayScaleObservation
from architecture.rl.env import ResizeObservation
from architecture.rl.env import FrameStack
from architecture.rl.network import MarioNet
from architecture.rl.memory import Memory
from architecture.rl.network import QValueEstimator
from architecture.rl.network import ValueEstimator
from architecture.rl.policy import PolicyUpdater
from architecture.rl.memory import BatchTransitionMemoryRecaller

from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros


class Orchestrator:

    def __init__(
            self,
            num_episodes: int = 101,
            log_frequency: int = 20,
            debug: bool = True,
    ) -> None:
        self.num_episodes = num_episodes

        self.save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.save_dir.mkdir(parents=True)

        self.log_frequency = log_frequency
        self.logger = MetricLogger(self.save_dir)

        self.debug = debug

    def train(self) -> Agent:
        env = self.initialize_env()
        agent = self.create_agent(env, self.save_dir)

        for e in range(self.num_episodes):
            self.run_loop(env, agent)

            self.logger.log_episode()
            if e % self.log_frequency == 0:
                self.logger.record(episode=e, step=agent.curr_step)

        return agent

    def run_loop(self, env, agent, train=True):
        state = env.reset()
        done = False
        while not done:
            if not train:
                env.render(mode='human')
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            if train:
                q, loss = agent.learn()
                self.logger.log_step(reward, loss, q)
                agent.cache(state, next_state, action, reward, done)

            state = next_state
            if done:
                break

    def create_agent(self, env, save_dir: Path) -> Agent:
        use_cuda = torch.cuda.is_available()

        env.reset()
        next_state, reward, done, info = env.step(action=0)
        if self.debug:
            print(f"{next_state.shape},\n {reward},\n {done},\n {info}\n")
        memory = Memory(max_size=100000, use_cuda=use_cuda)
        network = MarioNet(input_dim=next_state.shape, action_dim=env.action_space.n, save_dir=save_dir) \
            .to(device=torch.device('cuda' if use_cuda else 'cpu'))

        return Agent(
            save_dir=save_dir,
            network=network,
            memory=memory,
            action_selector=EpsilonGreedyActionSelector(
                action_dim=env.action_space.n,
                use_cuda=use_cuda,
                exploration_decay_rate=0.99999975,
                exploration_rate_min=0.1,
            ),
            policy_updaters=[
                PolicyUpdater(
                    mem_recaller=BatchTransitionMemoryRecaller(memory, batch_size=32),
                    loss_fn=torch.nn.SmoothL1Loss(),
                    optimizer=torch.optim.Adam(network.parameters(), lr=0.00025),
                    predictor=QValueEstimator(network=network, model_type='online'),
                    targeter=ValueEstimator(network=network, gamma=0.9)
                ),
            ],
        )

    def initialize_env(self):
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        env = self.apply_env_wrappers(env)
        return env

    def apply_env_wrappers(self, env):
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)

        env = JoypadSpace(env, [["right"], ["right", "A"]])
        return env

    def evaluate(self, agent: Agent) -> None:
        env = self.initialize_env()
        self.run_loop(env=env, agent=agent, train=False)


if __name__ == '__main__':
    orchestrator = Orchestrator(num_episodes=21)
    agent = orchestrator.train()

    orchestrator.evaluate(agent)
