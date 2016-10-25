import tensorflow as tf

class NetworkVariable:
    def __init__(self, shape):
        self.shape = shape
        self.input = tf.placeholder(dtype=tf.float32, shape=shape)
        self.variable = self.create_variable()
        self.assign_op = tf.assign(self.variable, self.input)

    def create_variable(self):
        raise NotImplementedError

    def get_assign_operation(self):
        return self.assign_op

    def get_variable(self):
        return self.variable

    def get_placeholder(self):
        return self.input

    def assign(self, source, session):
        op = self.get_assign_operation()
        op.eval(session=session, feed_dict={
                self.get_placeholder():
                           source.get_variable().eval(session=session)
            })



class WeightsVariable(NetworkVariable):
    def __init__(self, shape):
        NetworkVariable.__init__(self, shape)

    def create_variable(self):
        return self.weight_variable(self.shape)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.02)
        return tf.Variable(initial)

class BiasVariable(NetworkVariable):
    def __init__(self, shape):
        NetworkVariable.__init__(self, shape)

    def create_variable(self):
        return self.bias_variable(self.shape)

    def bias_variable(self, shape):
        initial = tf.zeros(shape=shape)
        return tf.Variable(initial)

