from typing import List
from typing import Tuple

from architecture.rl.action import ActionSelector
from architecture.rl.memory import Memory
from architecture.rl.network import Network
from architecture.rl.policy import PolicyUpdater


class AgentConfig:

    def __init__(
        self,
        save_every: int = 5e5,
        burnin: int = 1e4,
        learn_every: int = 3,
        sync_every: int = 1e4,
    ) -> None:
        self.save_every = save_every  # no. of experiences between saving Mario Net
        self.burnin = burnin  # min. experiences before training
        self.learn_every = learn_every  # no. of experiences between updates to Q_online
        self.sync_every = sync_every  # no. of experiences between Q_target & Q_online sync

    def should_sync(self, current_step: int) -> bool:
        return current_step % self.sync_every == 0

    def should_save(self, current_step: int) -> bool:
        return current_step % self.save_every == 0

    def should_learn(self, current_step: int) -> bool:
        return current_step < self.burnin or current_step % self.learn_every != 0


class Agent:

    def __init__(
        self,
        save_dir,
        network: Network,
        action_selector: ActionSelector,
        memory: Memory,
        policy_updaters: List[PolicyUpdater],
    ) -> None:
        self.save_dir = save_dir

        self.memory = memory

        self.net = network
        self.action_selector = action_selector
        self.policy_updaters = policy_updaters

        self.config = AgentConfig()

        self.curr_step = 0
        self.save_iteration = 0

    def act(self, state):
        action_idx = self.action_selector.select_action(state, self.net)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        self.memory.cache(state, next_state, action, reward, done)

    def update_network(self) -> Tuple[float, float]:
        td_estimate = 0.0
        loss = 0.0
        for updater in self.policy_updaters:
            estimate, policy_loss = updater.update()
            td_estimate += estimate.mean().item()
            loss += policy_loss.item()
        return td_estimate, loss

    def save(self):
        save_path = self.net.save(self.save_iteration)
        self.save_iteration += 1
        print(f"Network saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.config.should_sync(self.curr_step):
            self.net.sync_target_to_online()

        if self.config.should_save(self.curr_step):
            self.save()

        if self.config.should_learn(self.curr_step):
            return None, None

        td_est, loss = self.update_network()

        return td_est, loss
