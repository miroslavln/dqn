import numpy as np

class FrameBuffer:
  def __init__(self, args):
    self.history_length = args.history_length
    self.screen_size = (args.screen_height, args.screen_width)
    self.buffer = np.zeros([1, self.history_length, args.screen_height, args.screen_width], dtype=np.uint8)

  def add(self, screen):
    assert screen.shape == self.screen_size
    assert screen.dtype == np.uint8
    self.buffer = np.roll(self.buffer, 1, axis=1)
    self.buffer[0, 0] = screen
    assert np.count_nonzero(self.buffer[0, 0]) > 0

  def get_state(self):
    return self.buffer[0].transpose(1, 2, 0)

  def get_state_as_batch(self):
    return self.buffer.transpose(0, 2, 3, 1)

  def reset(self):
    self.buffer *= 0

'''
class args:
  def __init__(self):
      self.screen_height = 10
      self.screen_width = 10
      self.history_length = 3

buf = FrameBuffer(args())
ar = np.ndarray([10,10])
ar[0] = 1
buf.add(ar)
ar[0] = 2
buf.add(ar)
ar[0] = 3
buf.add(ar)
'''
