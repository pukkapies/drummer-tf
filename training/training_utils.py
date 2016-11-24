import numpy as np

class Patience(object):
    """
    Object to keep track of patience, and adjust the learning rate according to a prescribed list.
    When terminate method is called, it returns a list of 3 elements:
        1. Whether patience is reached
        2. If the cost obtained is a new best cost.
    """

    def __init__(self, args):
        self.best_cost = np.inf
        self.iterations = 0
        self.max_iterations = args.max_patience
        self.learning_rates_index = 0

    def update(self, cost):
        if cost < self.best_cost: # New best cost obtained, reset patience and save the cost
            self.iterations = 0
            self.best_cost = cost
            return False, True
        else:
            self.iterations += 1
            if self.iterations >= self.max_iterations:  # Patience reached, change the learning rate or terminate
                self.learning_rates_index += 1
                self.iterations = 0
                return True, False
            else:   # Nothing happened
                return False, False
