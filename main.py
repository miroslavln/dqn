import logging

from statistics import Statistics

logging.basicConfig(format='%(asctime)s %(message)s')

from environment import GymEnvironment
from agent import DqnAgent
from dqn import DeepQNetwork
from replay_memory import ReplayMemory
import random
import argparse
import sys

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("game", default="SpaceInvaders-v0", help="ROM bin file or env id such as Breakout-v0 if training with Open AI Gym.")
envarg.add_argument("--display_screen", type=str2bool, default=False, help="Display game screen during training and testing.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=300000, help="Maximum size of replay memory.")
memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--target_steps", type=int, default=5000, help="Copy main network to target network after this many steps.")
netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--exploration_rate_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
antarg.add_argument("--exploration_rate_test", type=float, default=0.05, help="Exploration rate used during testing.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--train_steps", type=int, default=10000, help="How many training steps per epoch.")
mainarg.add_argument("--test_steps", type=int, default=1000, help="How many testing steps after each epoch.")
mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
mainarg.add_argument("--start_epoch", type=int, default=0, help="Start from this epoch, affects exploration rate and names of saved snapshots.")
mainarg.add_argument("--play_games", type=int, default=0, help="How many games to play, suppresses training and testing.")
mainarg.add_argument("--load_weights", help="Load network from file.")
mainarg.add_argument("--save_weights_prefix", help="Save network to given file. Epoch and extension will be appended.")

comarg = parser.add_argument_group('Common')
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_level)

if args.random_seed:
  random.seed(args.random_seed)

# instantiate classes
env = GymEnvironment(args.game, args)
logger.info("Using Gym Environment")
net = DeepQNetwork(env.numActions(), args)
statistics = Statistics(net)
mem = ReplayMemory(args.replay_size, args)
agent = DqnAgent(env, mem, net, args, statistics=statistics)

if args.load_weights:
  logger.info("Loading weights from %s" % args.load_weights)
  net.load_weights(args.load_weights)

if args.play_games:
  logger.info("Playing for {} game(s)".format(args.play_games))
  agent.play(args.play_games)
  sys.exit()

for epoch in range(args.start_epoch, args.epochs):
  logger.info("Epoch #{}/{}".format(epoch + 1, args.epochs))

  if args.train_steps:
    logger.info(" Training for {} steps".format(args.train_steps))
    agent.train(args.train_steps, epoch)

    if args.save_weights_prefix:
      filename = args.save_weights_prefix
      logger.info("Saving weights to {}".format(filename))
      net.save_weights(filename)

  if args.test_steps:
    logger.info(" Testing for %{} steps".format(args.test_steps))
    agent.test(args.test_steps, epoch)

logger.info("All done")