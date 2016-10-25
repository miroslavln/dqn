import logging
import numpy as np
import random

logger = logging.getLogger()

class ReplayMemory:
    def __init__(self, size, args):
        self.size = size
        # preallocate memory
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.integer)
        self.states = np.empty((self.size, args.screen_height, args.screen_width, args.history_length), dtype=np.uint8)
        self.terminals = np.empty(self.size, dtype=np.bool)
        self.history_length = args.history_length
        self.state_shape = (args.screen_height, args.screen_width, args.history_length)
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0

        logger.info("Replay memory size: %d" % self.size)

    def add(self, action, reward, state, terminal):
        assert state.shape == self.state_shape, state.shape
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current, ...] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current += 1
        self.current %= self.size

    def getMinibatch(self):
        assert self.count > self.history_length
        non_terminals = np.where(self.terminals[:self.count] < 1)[0]
        indexes = np.random.choice(non_terminals, self.batch_size)

        poststate_indexes = (indexes + 1) % self.size
        actions = self.actions[poststate_indexes]
        rewards = self.rewards[poststate_indexes]
        terminals = self.terminals[poststate_indexes]
        prestates = self.states[indexes]
        poststates = self.states[poststate_indexes]

        return prestates, actions, rewards, poststates, terminals