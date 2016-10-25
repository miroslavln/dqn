class Statistics:
    def __init__(self, dqn):
        self.reset(0)
        self.dqn = dqn

    def update(self, action, reward, terminal, epsilon):
        self.reward += reward
        self.max_reward = max(self.max_reward, reward)
        self.min_reward = min(self.min_reward, reward)
        self.epsilon = epsilon

        if terminal:
            self.num_games += 1

    def reset(self, epoch):
        self.epoch = epoch
        self.num_games = 1
        self.reward = float(0)
        self.max_reward = 0
        self.min_reward = 0

    def display(self):
        print('Epoch {}, num_games {}, average reward {}, learning rate {}'.format(self.epoch,
                                                                                self.num_games,
                                                                                self.reward / self.num_games,
                                                                                self.epsilon))
        self.dqn.add_statistics(self.epoch, self.num_games, self.reward/self.num_games)