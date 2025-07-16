import logging
import os
import numpy as np
import tensorflow as tf

from vision_transformer import VisionTransformer
from network_variable import WeightsVariable, BiasVariable, NetworkVariable

logger = logging.getLogger()

CONV_LAYER ='conv'

class DeepQNetwork:
    def __init__(self, num_actions, args):
        self.graph = tf.Graph()
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

        self.model_network = {}
        self.target_network = {}

        self.build_graph()

        self.session = tf.compat.v1.Session(graph=self.graph)
        self.train_writer = tf.compat.v1.summary.FileWriter('logs/train', self.session.graph)
        self.session.run(self.initCmd)

    def create_network(self, x, model, name):
        with tf.variable_scope(name):
            return VisionTransformer(self.num_actions)(x)

    def build_graph(self):
        with self.graph.as_default():
            self.create_models()
            self.define_optimizer()

            self.define_summary_operations()

            self.saver = tf.compat.v1.train.Saver()
            self.initCmd = tf.compat.v1.global_variables_initializer()


    def create_models(self):
        self.batch = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 84, 84, self.history_length], name='s')
        normalized_batch = tf.divide(self.batch, 255)
        self.model = self.create_network(normalized_batch, self.model_network, 'pre_q')
        self.target_model = self.create_network(normalized_batch, self.target_network, 'post_q')


    def define_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.actions = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None], name='actions')
            self.targets = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='targets')

            actions = tf.one_hot(self.actions, self.num_actions, 1.0, 0)
            self.q_values = tf.reduce_sum(self.model * actions, reduction_indices=1)

            delta = self.targets - self.q_values
            clipped_delta = tf.clip_by_value(delta, -self.clip_error, self.clip_error, name='clipped_delta')

            self.loss = tf.reduce_mean(tf.square(clipped_delta), name='loss')
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.95, momentum=0.95, epsilon=0.01).minimize(self.loss)

    def define_summary_operations(self):
        with tf.variable_scope('Summary'):
            self.define_epoch_summary()
            self.define_training_summary()
            self.define_image_summary()

    def define_training_summary(self):
        ave_q = tf.compat.v1.summary.scalar('average q value', tf.reduce_mean(self.q_values))

        avg_q = tf.reduce_mean(self.model, 0)
        q_summary = [tf.compat.v1.summary.histogram('q/{}'.format(idx), avg_q[idx]) for idx in xrange(self.num_actions)]

        self.train_summary = tf.compat.v1.summary.merge([ave_q]+q_summary)


    def define_epoch_summary(self):
        self.num_games = tf.compat.v1.placeholder(dtype=tf.int32)
        games_summary = tf.compat.v1.summary.scalar('num games/epoch', self.num_games)

        self.average_reward = tf.compat.v1.placeholder(dtype=tf.float32)
        reward_summary = tf.compat.v1.summary.scalar('average reward/game', self.average_reward)

        self.epoch_summary = tf.compat.v1.summary.merge([games_summary, reward_summary])

    def define_image_summary(self):
        batch_images = tf.compat.v1.summary.image("convolution image", self.model_network[CONV_LAYER][:,:,:, :1], max_images=32)
        self.image_summary = tf.compat.v1.summary.merge([batch_images])

    def assign_model_to_target(self):
        for name in self.model_network.keys():
            if not isinstance(self.target_network[name], NetworkVariable):
                continue
            self.target_network[name].assign(self.model_network[name], session=self.session)

    def save_weights(self, file_name):
        save_path = self.saver.save(self.session, file_name)
        logger.info("Model saved in file: %s" % save_path)

    def load_weights(self, file_name):
        logger.info("Loading models saved in file: %s" % file_name)
        self.saver.restore(self.session, file_name)

    def train(self, minibatch, epoch):
        s, actions, rewards, s_prime, terminals = minibatch

        postq = self.get_q_values(s_prime, self.target_model)
        max_postq = np.max(postq, axis=1)

        rewards = np.clip(rewards, self.min_reward, self.max_reward)

        '''
        for i, action in enumerate(actions):
            if terminals[i]:
                target[i, action] = float(rewards[i])
            else:
                target[i, action] = float(rewards[i]) + self.discount_rate * max_postq[i]
        '''
        target = rewards + (1.0 - terminals) * (self.discount_rate * max_postq)

        feed_dict = {self.batch: s, self.actions: actions, self.targets: target}
        _, train_summaryStr, image_summaryStr  = self.session.run([self.optimizer, self.train_summary, self.image_summary], feed_dict=feed_dict)
        self.train_writer.add_summary(train_summaryStr, self.total_training_steps)

        if self.total_training_steps % self.target_steps == 0:
            self.assign_model_to_target()

        self.total_training_steps += 1

    def get_q_values(self, state, model):
        feed_dict = {self.batch: state}

        return self.session.run(model, feed_dict=feed_dict)

    def predict(self, state):
        return self.get_q_values(state, self.model)

    def add_statistics(self, epoch, num_games, average_reward):
        epoch_summary_str = self.epoch_summary.eval(session=self.session, feed_dict={
            self.num_games: num_games,
            self.average_reward: average_reward})

        self.train_writer.add_summary(epoch_summary_str, epoch)
        self.train_writer.flush()


