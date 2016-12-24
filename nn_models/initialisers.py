import tensorflow as tf
import numpy as np


def wbVars_XavierReLU(fan_in: int, fan_out: int):
    """Helper to initialize weights and biases, via He's adaptation
    of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
    """
    # (int, int) -> (tf.Variable, tf.Variable)
    stddev = tf.cast((2 / fan_in ) **0.5, tf.float32)

    initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
    initial_b = tf.zeros([fan_out])

    return (tf.Variable(initial_w, trainable=True, name="weights"),
            tf.Variable(initial_b, trainable=True, name="biases"))


def wbVars_Xavier(fan_in: int, fan_out: int):
    """Helper to initialize weights and biases, via Xavier init:
    http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    # (int, int) -> (tf.Variable, tf.Variable)
    init_boundary = np.sqrt(6. / (fan_in + fan_out))

    weights = (tf.Variable(tf.random_uniform([fan_in, fan_out], minval=-init_boundary, maxval=init_boundary),
                           trainable=True, name="weights"))
    biases = tf.Variable(tf.zeros([fan_out]), trainable=True, name="biases")

    return weights, biases