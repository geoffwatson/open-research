from architecture.rl.memory import MemoryRecaller
from architecture.rl.network import NetworkCaller


class PolicyUpdater:

    def __init__(
            self,
            mem_recaller: MemoryRecaller,
            loss_fn,
            optimizer,
            predictor: NetworkCaller,
            targeter: NetworkCaller,
    ) -> None:
        self.mem_recaller = mem_recaller
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.predictor = predictor
        self.targeter = targeter

    def update(self):
        tensors = self.mem_recaller.recall()

        estimate = self.predictor.call(tensors)
        target = self.targeter.call(tensors)

        loss = self.loss_fn(estimate, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return estimate, loss
