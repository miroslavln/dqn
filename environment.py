import gymnasium as gym
import cv2
import ale_py

class Environment:
  def __init__(self):
    pass

  def numActions(self):
    # Returns number of actions
    raise NotImplementedError

  def restart(self):
    # Restarts environment
    raise NotImplementedError

  def act(self, action):
    # Performs action and returns reward
    raise NotImplementedError

  def getScreen(self):
    # Gets current game screen
    raise NotImplementedError

  def isTerminal(self):
    # Returns if game is done
    raise NotImplementedError


class GymEnvironment(Environment):
  def __init__(self, env_id, args):
    self.gym = gym.make(env_id)
    self.obs = None
    self.terminal = None
    self.display = args.display_screen
    self.dims = (args.screen_width, args.screen_height)

  def numActions(self):
    assert isinstance(self.gym.action_space, gym.spaces.discrete.Discrete)
    return self.gym.action_space.n

  def restart(self):
    self.obs = self.gym.reset()
    self.terminal = False

  def act(self, action):
    self.obs, reward, self.terminal, _, _ = self.gym.step(action)
    if self.display:
      self.gym.render()
    return reward

  def getScreen(self):
    assert self.obs is not None
    screen = cv2.cvtColor(self.obs, cv2.COLOR_BGR2GRAY)
    return cv2.resize(screen, self.dims)

  def isTerminal(self):
    assert self.terminal is not None
    return self.terminal
