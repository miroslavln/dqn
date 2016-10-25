import random
import logging
import numpy as np
from tqdm import tqdm

from frame_buffer import FrameBuffer

logger = logging.getLogger()

class DqnAgent(object):
    def __init__(self, environment, replay_memory, dqn, args, statistics=None):
        self.env = environment
        self.net = dqn
        self.mem = replay_memory
        self.statistics = statistics
        self.buf = FrameBuffer(args)
        self.num_actions = self.env.numActions()
        self.history_length = args.history_length

        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_steps

        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end
        self.exploration_decay_steps = args.train_steps * args.epochs / 3

    def _get_exploration_rate(self):
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * (
            self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
        else:
            return self.exploration_rate_end

    def step(self, exploration_rate):
        action = self._pick_action(exploration_rate)

        reward, new_state, terminal = self._perform_action(action)

        if terminal:
            logger.debug("Terminal state, restarting")
            self._restart()

        return action, reward, new_state, terminal

    def _perform_action(self, action):
        reward, terminal, screen = 0, False, None
        for i in xrange(self.history_length):
            if not terminal:
                reward += self.env.act(action)
                screen = self.env.getScreen()
                terminal = self.env.isTerminal()

            self.buf.add(screen)

        return reward, self.buf.get_state(), terminal

    def _pick_action(self, exploration_rate):
        if random.random() < exploration_rate:
            action = random.randrange(self.num_actions)
            logger.debug("Random action = %d" % action)
        else:
            state = self.buf.get_state_as_batch()
            qvalues = self.net.predict(state)
            assert len(qvalues[0]) == self.num_actions

            action = np.argmax(qvalues[0])
            logger.debug("Chosen action = %d" % action)
        return action

    def train(self, train_steps, epoch=0):
        logger.info("Training. Exploration rate {}".format(self._get_exploration_rate()))
        for i in tqdm(xrange(train_steps)):
            action, reward, new_state, terminal = self.step(self._get_exploration_rate())
            self.mem.add(action, reward, new_state, terminal)

            if self.mem.count > self.mem.batch_size:
                self._train_network(epoch)
                self.total_train_steps += 1

    def _train_network(self, epoch):
        minibatch = self.mem.getMinibatch()
        self.net.train(minibatch, epoch)

    def _restart(self):
        self.env.restart()
        for i in xrange(self.history_length):
            reward = self.env.act(np.random.choice(self.num_actions))
            screen = self.env.getScreen()
            terminal = self.env.isTerminal()
            self.buf.add(screen)

    def play(self, num_games):
        self._restart()
        for i in xrange(num_games):
            terminal = False
            while not terminal:
                action, reward, new_state, terminal = self.step(self.exploration_rate_test)
                self.mem.add(action, reward, new_state, terminal)

    def test(self, test_steps, epoch=0):
        if self.statistics:
            self.statistics.reset(epoch)

        self._restart()
        for i in tqdm(xrange(test_steps)):
            action, reward, new_state, terminal = self.step(self.exploration_rate_test)
            if self.statistics:
                self.statistics.update(action, reward, terminal, self._get_exploration_rate())

        if self.statistics:
            self.statistics.display()





