import numpy as np
import random

class LinearModel:
    def __init__(self, alpha, convergence_j_threshold, 
            consecutive_j_thresh_cnt, max_iter, seed = None, 
            reg_type = None, reg_lambda = 0, l1_prop = 0):
        self.alpha = alpha # learning rate
        self.convergence_j_threshold = convergence_j_threshold
        # consecutive times as you occasionally get lucky
        self.consecutive_j_thresh_cnt = consecutive_j_thresh_cnt
        self.max_iter = max_iter
        self.seed = seed
        VALID_REG_TYPES = [None, 'l1', 'lasso','l2', 'ridge', 'elastic_net']
        if reg_type not in VALID_REG_TYPES:
            raise ValueError('reg_type must be one of {}'.format(VALID_REG_TYPES))
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.l1_prop = l1_prop
        self.b_hat = np.empty((0, 0))
        self.X = np.empty((0, 0))
        self.y = np.empty((0, 0))
        self.fitted = False
    
    # check converged criteria met
    def _converged(self, i, low_j_cnt):
        return i >= self.max_iter or \
               low_j_cnt >= self.consecutive_j_thresh_cnt
    
    def _j_reg_term(self):
        if not self.reg_type:
            return 0
        elif self.reg_type in ['l1', 'lasso']:
            return self.reg_lambda * sum(abs(self.b_hat))
        elif self.reg_type in ['l2', 'ridge']:
            return self.reg_lambda / 2 * sum(self.b_hat**2) / self.X.shape[0]
        elif self.reg_type in ['elastic_net']:
            return self.reg_lambda * (self.l1_prop * sum(np.absolute(self.b_hat)) + \
                (1 - self.l1_prop) / 2 * sum(self.b_hat**2) / self.X.shape[0])
    
    def _b_hat_reg_term(self):
        if not self.reg_type:
            return 0
        elif self.reg_type in ['l1', 'lasso']:
            return self.reg_lambda * self.b_hat / abs(self.b_hat)
        elif self.reg_type in ['l2', 'ridge']:
            return self.reg_lambda * self.b_hat / self.X.shape[0]
        elif self.reg_type in ['elastic_net']:
            return self.reg_lambda * (self.l1_prop * self.b_hat / abs(self.b_hat) + \
                (1 - self.l1_prop) * self.b_hat / self.X.shape[0])
        
    def __repr__(self):
        return 'LinearModel(\nalpha (gradient descent step size) = {}\n\
        reg_type = {}\nreg_lambda = {}\nl1_prop = {}\n\
        convergence_j_threshold = {}\nconsecutive_j_thresh_cnt = {}\nmax_iter = {}\n)'\
            .format(self.alpha, self.reg_type, self.reg_lambda, self.l1_prop, \
                self.convergence_j_threshold, self.consecutive_j_thresh_cnt, \
                self.max_iter)

class LinearRegression(LinearModel):
    def fit(self, X, y):
        random.seed(self.seed) # self.seed=None means use system time

        # INITIALIZE
        row_cnt = X.shape[0]
        # add constant term
        self.X, self.y = np.hstack([np.ones(row_cnt)[:, np.newaxis], X]), y
        variable_cnt = self.X.shape[1] # after adding constant term
        self.b_hat = np.ones(variable_cnt)
        self.j_each_iteration = [] # cost function
        # convergence criteria 1 - cost function less than threshold
        consecutive_low_j_current_cnt = 0
        # convergence criteria 2 - reach max iterations
        current_iter = 0

        # RUN batch gradient descent
        # converged (i.e., do not run) if max iteration OR convergence criteria are met
        while not self._converged(current_iter, consecutive_low_j_current_cnt): 
            # calculate numbers
            y_hat = self.X @ self.b_hat
            diff = y_hat - self.y
            j = np.mean(diff**2) / 2 + self._j_reg_term()
            
            # update
            self.j_each_iteration.append(j)
            self.b_hat -= self.alpha * np.mean(diff[:, np.newaxis] * self.X, axis = 0) + \
                self._b_hat_reg_term()
            current_iter += 1
            consecutive_low_j_current_cnt = consecutive_low_j_current_cnt + 1 \
                if abs(j) < self.convergence_j_threshold else 0
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise AssertionError('Object not fitted, no b_hat values')
        sample_cnt = X.shape[0]
        y_hat = np.hstack([np.ones(sample_cnt)[:, np.newaxis], X]) @ \
                     self.b_hat
        return y_hat
        
class LogisticRegression(LinearModel):
    def fit(self, X, y):
        random.seed(self.seed) # self.seed=None means use system time

        # INITIALIZE
        row_cnt = X.shape[0]
        # add constant term
        self.X, self.y = np.hstack([np.ones(row_cnt)[:, np.newaxis], X]), y
        variable_cnt = self.X.shape[1] # after adding constant term
        self.b_hat = np.ones(variable_cnt)
        self.j_each_iteration = [] # cost function
        # convergence criteria 1 - cost function less than threshold
        consecutive_low_j_current_cnt = 0
        # convergence criteria 2 - reach max iterations
        current_iter = 0

        # RUN batch gradient descent
        # converged (i.e., do not run) if max iteration OR convergence criteria are met
        while not self._converged(current_iter, consecutive_low_j_current_cnt): 
            # calculate numbers
            y_hat = (np.exp(-1 * self.X @ self.b_hat) + 1)**-1
            diff = y_hat - self.y
            j = np.mean(self.y * np.log1p(y_hat) + \
                (1 - self.y) * np.log1p(1 - y_hat)) + \
                self._j_reg_term()
            
            # update
            self.j_each_iteration.append(j)
            self.b_hat -= self.alpha * np.sum(diff[:, np.newaxis] * self.X, axis = 0) \
                + self._b_hat_reg_term()
            current_iter += 1
            consecutive_low_j_current_cnt = consecutive_low_j_current_cnt + 1 \
                if abs(j) < self.convergence_j_threshold else 0
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise AssertionError('Object not fitted, no b_hat values')
        sample_cnt = X.shape[0]
        X = np.hstack([np.ones(sample_cnt)[:, np.newaxis], X])
        y_hat = (np.exp(-1 * X @ self.b_hat) + 1)**-1
        return y_hat

if __name__ == '__main__':
    pass