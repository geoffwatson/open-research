import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, save_dir: Path, **kwargs):
        super(Network, self).__init__()
        self.save_dir = save_dir

        self.online = nn.ModuleDict(self.build_online_models(**kwargs))

        self.target = {key: copy.deepcopy(value) for key, value in self.online.items()}
        for _, target_network in self.target.items():
            # Q_target parameters are frozen.
            for p in target_network.parameters():
                p.requires_grad = False

    def sync_target_to_online(self) -> None:
        for key in self.online.keys():
            self.target[key].load_state_dict(self.online[key].state_dict())

    def save(self, checkpoint_number: int) -> Path:
        save_path = (
                self.save_dir / f"{self.get_name()}_{checkpoint_number}.chkpt"
        )
        torch.save(self.net.state_dict(), save_path)
        return save_path

    def build_online_models(self, **kwargs) -> Dict[str, nn.Module]:
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError


class MarioNet(Network):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_online_models(self, **kwargs) -> Dict[str, nn.Module]:
        c, _, _ = kwargs['input_dim']
        return {
            'main': nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, kwargs['action_dim']),
            )
        }

    def forward(self, input, model):
        if model == "online":
            return self.online['main'](input)
        elif model == "target":
            return self.target['main'](input)


class NetworkCaller:

    def __init__(self, network):
        self.network = network

    def call(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class QValueEstimator(NetworkCaller):

    def __init__(self, network, model_type: str = "online") -> None:
        super(QValueEstimator, self).__init__(network)
        self.model_type = model_type

    def call(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = np.arange(0, input_dict['action'].Size()[0])
        # Get selected action value from each batch prediction
        return self.network(input_dict['state'], model=self.model_type)[batch, input_dict['action']]


class ValueEstimator(NetworkCaller):

    def __init__(self, network, gamma) -> None:
        super(ValueEstimator, self).__init__(network)
        self.gamma = gamma
        self.q_value_estimator = QValueEstimator(network, "target")

    def call(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            next_state_q = self.network(input_dict['next_state'], model="online")
            input_dict['action'] = torch.argmax(next_state_q, dim=1)
            next_q = self.q_value_estimator.call(input_dict)
            return (input_dict['reward'] + (1 - input_dict['done'].float()) * self.gamma * next_q).float()
