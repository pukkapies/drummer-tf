import numpy as np


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

