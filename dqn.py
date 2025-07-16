import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vision_transformer import VisionTransformer

logger = logging.getLogger()

class DeepQNetwork(nn.Module):
    def __init__(self, num_actions, args):
        super(DeepQNetwork, self).__init__()
        self.num_actions = num_actions
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.history_length = args.history_length
        self.clip_error = args.clip_error
        self.min_reward = args.min_reward
        self.max_reward = args.max_reward
        self.learning_rate = args.learning_rate
        self.target_steps = args.target_steps
        self.total_training_steps = args.start_epoch * args.train_steps

        self.model = VisionTransformer(self.num_actions)
        self.target_model = VisionTransformer(self.num_actions)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=self.learning_rate,
            alpha=0.95,
            eps=0.01,
            momentum=0.95,
        )
        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter('logs/train')

    def save_weights(self, file_name):
        torch.save(self.model.state_dict(), file_name)
        logger.info("Model saved in file: %s" % file_name)

    def load_weights(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.target_model.load_state_dict(self.model.state_dict())
        logger.info("Loading models saved in file: %s" % file_name)

    def train(self, minibatch, epoch):
        s, actions, rewards, s_prime, terminals = minibatch
        s = torch.from_numpy(s).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        s_prime = torch.from_numpy(s_prime).float()
        terminals = torch.from_numpy(terminals).float()

        postq = self.get_q_values(s_prime, self.target_model)
        max_postq = torch.max(postq, dim=1).values
        rewards = torch.clamp(rewards, self.min_reward, self.max_reward)
        target = rewards + (1.0 - terminals) * (self.discount_rate * max_postq)

        q_values = self.get_q_values(s, self.model)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_values_for_actions, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_error)
        self.optimizer.step()

        self.writer.add_scalar('Loss/train', loss, self.total_training_steps)

        if self.total_training_steps % self.target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.total_training_steps += 1

    def get_q_values(self, state, model):
        return model(state)

    def predict(self, state):
        state = torch.from_numpy(state).float()
        return self.get_q_values(state, self.model).detach().numpy()

    def add_statistics(self, epoch, num_games, average_reward):
        self.writer.add_scalar('Epoch/num_games', num_games, epoch)
        self.writer.add_scalar('Epoch/average_reward', average_reward, epoch)
        self.writer.flush()
