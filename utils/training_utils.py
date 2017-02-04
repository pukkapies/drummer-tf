import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients import AggregationMethod


class Patience(object):
    """
    Object to keep track of patience, and adjust the learning rate according to a prescribed list.
    When terminate method is called, it returns a list of 3 elements:
        1. Whether patience is reached
        2. If the cost obtained is a new best cost.
        3. If a plateau was reached
    """

    def __init__(self, args):
        self.best_cost = np.inf
        self.best_cost_step = 0
        self.global_step = 0
        self.iterations = 0
        self.max_iterations = args.max_patience
        self.learning_rates_index = 0
        self.plateau = PlateauDetector(args, self.global_step)

    def update(self, cost):
        self.global_step += 1
        if cost < self.best_cost: # New best cost obtained, reset patience and save the cost
            self.best_cost = cost
            self.best_cost_step = self.global_step
            self.iterations = 0
            return False, True, False
        else:
            self.iterations += 1
            if self.iterations >= self.max_iterations:  # Patience reached, change the learning rate or terminate
                self.learning_rates_index += 1
                self.iterations = 0
                return True, False, False
            # elif self.plateau.detect_plateau(self.global_step, self.best_cost_step, cost, self.best_cost):
            #     return False, False, True
            else:   # Nothing happened
                return False, False, False


#TODO: This is totally broken. Figure out a better way of doing this
class PlateauDetector(object):
    """
    Object to determine if a plateau has been reached and therefore terminate the training. args contains
    a plateau_tol attribute, which is a 2-element list [n_iteration, minimum_cost_decrease]. If n_iteration's
    go by with a decrease in cost of less than minimum_cost_decrease, training will terminate.
    """

    def __init__(self, args, step):
        # self.num_plateau_iterations = args.plateau_tol[0]
        # self.minimum_cost_decrease = args.plateau_tol[1]
        # self.reference_step = step
        pass

    def detect_plateau(self, global_step, best_cost_step, current_cost, best_cost):
        if (global_step - best_cost_step >= self.num_plateau_iterations) \
                    and (best_cost - current_cost < self.minimum_cost_decrease):
            return True
        else:
            return False


class GradientAccumulator(object):
    """
    Accumulates the gradients for trainable variables to allow training steps to be broken into several iterations,
    when the minibatch is too big to fit on the GPU.
    """

    def __init__(self, sess, grads_and_vars, optimizer):
        """
        Initialiser
        :param gradients: List of (gradient, variable) tuples as returned by the optimizer.compute_gradients op
        """
        self.sess = sess
        self._gradients_and_variables = grads_and_vars
        self._optimizer = optimizer

        # The gradients returned from compute_gradients is a list of (gradient tensor, Variable object) tuples
        self._var_to_grad = dict((var.name, grad) for grad, var in self._gradients_and_variables)
        self._varname_to_var = dict((var.name, var) for _, var in self._gradients_and_variables)
        self._var_keys = sorted(self._varname_to_var.keys())
        self._var_to_accum_grad = self._initialize_gradient_dict()

    def initialize_internal_variables(self):
        for key in self._var_to_accum_grad:
            self.sess.run(tf.initialize_variables([self._var_to_accum_grad[key]]))

    def _initialize_gradient_dict(self):
        var_to_acc_grad = {}
        print("Constructing the gradient accumulator dict")
        for k in self._var_keys:
            with tf.variable_scope('grad_acc'):
                grad = self._var_to_grad[k]
                print(k, grad.get_shape())
                acc = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='accumulated_gradient')
                var_to_acc_grad[k] = acc
        return var_to_acc_grad

    def clear_gradients(self):
        """Returns a list of ops to reset all stored gradients to zero"""
        print("Clearing the gradients in the accumulator dict")
        clear_ops = []
        for k in self._var_keys:
            grad = self._var_to_grad[k]
            clear_ops.append(self._var_to_accum_grad[k].assign(tf.zeros(grad.get_shape())))
        return clear_ops

    def update_gradients_ops(self):
        """Returns a list of ops to update all gradients in the _var_to_accum_grad dict"""
        print("Updating gradients")
        update_ops = []
        for k in self._var_keys:
            grad = self._var_to_grad[k]
            update_ops.append(self._var_to_accum_grad[k].assign_add(grad))
        return update_ops

    def cumulative_gradient_list(self):
        """Returns a list of (grad, var) to be used in the optimizer.apply_gradients method"""
        var_grad_list = []
        for k in self._var_keys:
            grad = self._var_to_accum_grad[k]
            var = self._varname_to_var[k]
            var_grad_list.append((grad, var))
        return var_grad_list

def setup_VAE_training_ops(sess, learning_rate, cost, cost_no_KL, global_step):
    with tf.name_scope("Adam_optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()

        # Set AggregationMethod to try to avoid crash when computing all gradients simultaneously
        grads_and_vars = optimizer.compute_gradients(cost, tvars,
                                                          aggregation_method=AggregationMethod.EXPERIMENTAL_TREE)
        clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                   for grad, tvar in grads_and_vars]

        grads_and_vars_no_KL = optimizer.compute_gradients(cost_no_KL, tvars,
                                                                aggregation_method=AggregationMethod.EXPERIMENTAL_TREE)
        clipped_no_KL = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                         for grad, tvar in grads_and_vars_no_KL]

        gradient_acc = GradientAccumulator(sess, clipped, optimizer)
        gradient_acc_no_KL = GradientAccumulator(sess, clipped_no_KL, optimizer)

        apply_gradients_op = optimizer.apply_gradients(gradient_acc.cumulative_gradient_list(),
                                                            global_step=global_step, name='minimize_cost')
        apply_gradients_op_no_KL = optimizer.apply_gradients(gradient_acc_no_KL.cumulative_gradient_list(),
                                                                  global_step=global_step,
                                                                  name='minimize_cost_no_KL')

        # Initialize all variables used in the optimizer
        for var in tf.all_variables():
            if var in tf.get_collection(tf.GraphKeys.VARIABLES, scope='Adam_optimizer'):
                sess.run(tf.initialize_variables([var]))
    return optimizer, gradient_acc, gradient_acc_no_KL, apply_gradients_op, apply_gradients_op_no_KL

def setup_AE_training_ops(sess, learning_rate, cost, global_step):
    with tf.name_scope("Adam_optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()

        # Set AggregationMethod to try to avoid crash when computing all gradients simultaneously
        grads_and_vars = optimizer.compute_gradients(cost, tvars,
                                                          aggregation_method=AggregationMethod.EXPERIMENTAL_TREE)
        clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                   for grad, tvar in grads_and_vars]

        gradient_acc = GradientAccumulator(sess, clipped, optimizer)

        apply_gradients_op = optimizer.apply_gradients(gradient_acc.cumulative_gradient_list(),
                                                            global_step=global_step, name='minimize_cost')

        # Initialize all variables used in the optimizer
        for var in tf.all_variables():
            if var in tf.get_collection(tf.GraphKeys.VARIABLES, scope='Adam_optimizer'):
                sess.run(tf.initialize_variables([var]))
    return optimizer, gradient_acc, apply_gradients_op


class TrainingLog(object):

    def __init__(self, analysis_dir):
        self.total_cost_history = np.array([])
        self.reconstruction_cost_history = np.array([])
        self.analysis_dir = analysis_dir
        if self.analysis_dir[-1] != '/': self.analysis_dir += '/'

    def load_costs_from_file(self, global_step):
        self.total_cost_history = np.load(self.analysis_dir + 'total_cost.npy')[:global_step]
        self.reconstruction_cost_history = np.load(self.analysis_dir + 'reconstruction_cost.npy')[:global_step]

    def update_costs(self, costs_dict):
        self.total_cost_history = np.hstack((self.total_cost_history,
                                             np.array([float(costs_dict['total_cost_history'])])))
        self.reconstruction_cost_history = np.hstack((self.reconstruction_cost_history,
                                                      np.array([float(costs_dict['reconstruction_cost_history'])])))
