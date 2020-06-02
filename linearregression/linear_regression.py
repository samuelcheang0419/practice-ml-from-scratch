import numpy as np
import random
from ..utility import timer

class LinearRegressionWithSGD: 
    def __init__(self, alpha, convergence_se_threshold, 
            consecutive_se_thresh_cnt, max_iter, seed = None): 
        self.alpha = alpha # learning rate
        self.convergence_se_threshold = convergence_se_threshold
        # consecutive times as you occasionally get lucky
        self.consecutive_se_thresh_cnt = consecutive_se_thresh_cnt
        self.max_iter = max_iter 
        self.seed = seed

    @timer
    def fit(self, X, y): 
        random.seed(self.seed) # self.seed=None means use system time

        # INITIALIZE
        row_cnt = X.shape[0]
        # add constant term
        self.X, self.y = np.hstack([np.ones(row_cnt)[:, np.newaxis], X]), y
        variable_cnt = self.X.shape[1] # after adding constant term
        b_hat = np.zeros(variable_cnt)
        self.b_hat_each_iteration = np.empty((0, variable_cnt))
        self.se_each_iteration = [] # squared error
        self.diff_each_iteration = []
        # convergence criteria 1 - squared error less than threshold
        se = float('inf')
        consecutive_low_se_current_cnt = 0
        # convergence criteria 2 - reach max iterations
        current_iter = 0

        # check criteria met to run SGD
        def run_sgd(i, low_se_cnt):
            return i < self.max_iter and \
                   low_se_cnt < self.consecutive_se_thresh_cnt

        # RUN stochastic gradient descent (no mini-batch)
        # run if both max iteration and convergence criteria are not met
        while run_sgd(current_iter, consecutive_low_se_current_cnt): 
            # calculate numbers
            i = random.randint(0, row_cnt - 1)
            X_i, y_i = self.X[i, :], self.y[i]
            y_hat_i = X_i @ b_hat[:, np.newaxis]
            diff = y_hat_i - y_i
            se = diff**2
            
            # update
            self.b_hat_each_iteration = np.vstack([self.b_hat_each_iteration, b_hat])
            self.se_each_iteration.append(se[0])
            self.diff_each_iteration.append(diff[0])
            b_hat -= self.alpha * (y_hat_i - y_i) * X_i
            current_iter += 1
            if se < self.convergence_se_threshold: 
                consecutive_low_se_current_cnt += 1
            else: 
                consecutive_low_se_current_cnt = 0
        self.b_hat = b_hat
        return self

if __name__ == '__main__':
    pass