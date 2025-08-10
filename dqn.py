"""PyTorch implementation of Deep Q-Network.

This module replaces the original TensorFlow based implementation with a
minimal PyTorch version.  The interface of :class:`DeepQNetwork` remains the
same so that the rest of the project can interact with it without changes.

The network architecture matches the one used in the original DQN paper:

* 3 convolutional layers
* 2 fully connected layers

The class keeps two copies of the network (``model`` and ``target_model``).
The target network is periodically synchronised with ``model`` to stabilise
training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Small convolutional network used by :class:`DeepQNetwork`."""

    def __init__(self, history_length: int, num_actions: int) -> None:
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(history_length, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - tiny wrapper
        # Inputs are expected to be in [0, 255]; normalise to [0, 1].
        x = x / 255.0
        return self.net(x)


@dataclass
class DeepQNetwork:
    """Wraps two :class:`QNetwork` instances and handles optimisation."""

    num_actions: int
    args: any

    def __post_init__(self) -> None:
        self.batch_size = self.args.batch_size
        self.discount_rate = self.args.discount_rate
        self.history_length = self.args.history_length

        self.clip_error = self.args.clip_error
        self.min_reward = self.args.min_reward
        self.max_reward = self.args.max_reward

        self.learning_rate = self.args.learning_rate
        self.target_steps = self.args.target_steps
        self.total_training_steps = self.args.start_epoch * self.args.train_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create online and target networks
        self.model = QNetwork(self.history_length, self.num_actions).to(self.device)
        self.target_model = QNetwork(self.history_length, self.num_actions).to(self.device)
        self.assign_model_to_target()

        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=self.learning_rate,
            alpha=0.95,
            momentum=0.95,
            eps=0.01,
        )

    # ------------------------------------------------------------------ helpers
    def _prepare_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert a batch of states from numpy to a torch tensor.

        The environment stores states in ``(batch, H, W, C)`` format while
        PyTorch expects ``(batch, C, H, W)``.
        """

        state = torch.from_numpy(np.transpose(state, (0, 3, 1, 2))).float()
        return state.to(self.device)

    # ----------------------------------------------------------------- interface
    def train(self, minibatch: Tuple[np.ndarray, ...], epoch: int) -> None:
        """Performs one optimisation step using a minibatch of samples."""

        s, actions, rewards, s_prime, terminals = minibatch

        state = self._prepare_state(s)
        next_state = self._prepare_state(s_prime)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(np.clip(rewards, self.min_reward, self.max_reward)).float().to(self.device)
        terminals = torch.from_numpy(terminals.astype(np.float32)).to(self.device)

        with torch.no_grad():
            next_q = self.target_model(next_state).max(1)[0]
            target = rewards + (1.0 - terminals) * (self.discount_rate * next_q)

        q_values = self.model(state)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        delta = target - q_selected
        clipped_delta = torch.clamp(delta, -self.clip_error, self.clip_error)
        loss = (clipped_delta ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_training_steps += 1
        if self.total_training_steps % self.target_steps == 0:
            self.assign_model_to_target()

    # ---------------------------------------------------------------- prediction
    def get_q_values(self, state: np.ndarray, model: nn.Module | None = None) -> np.ndarray:
        """Returns Q-values for ``state`` using ``model`` (or ``self.model``)."""

        model = model or self.model
        with torch.no_grad():
            tensor_state = self._prepare_state(state)
            q = model(tensor_state).cpu().numpy()
        return q

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.get_q_values(state, self.model)

    # ---------------------------------------------------------------- maintenance
    def assign_model_to_target(self) -> None:
        """Copies parameters from the online network to the target network."""

        self.target_model.load_state_dict(self.model.state_dict())

    def save_weights(self, file_name: str) -> None:
        torch.save(self.model.state_dict(), file_name)
        logger.info("Model saved in file: %s", file_name)

    def load_weights(self, file_name: str) -> None:
        logger.info("Loading models saved in file: %s", file_name)
        state_dict = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.assign_model_to_target()

    # ---------------------------------------------------------------- statistics
    def add_statistics(self, epoch: int, num_games: int, average_reward: float) -> None:  # noqa: D401 - part of public API
        """Compatibility stub used by :class:`statistics.Statistics`.

        TensorFlow implementation logged data to TensorBoard; for the PyTorch
        port we currently do not log anything, but keeping the method preserves
        the public API.
        """

        # Intentionally left blank – no logging implementation.
        return None


__all__ = ["DeepQNetwork"]

