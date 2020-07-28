import numpy as np
import random

class CollaborativeFiltering:
    def __init__(self, alpha, convergence_j_threshold, max_iter, n_features, 
                 seed = None):
        self.alpha = alpha
        self.convergence_j_threshold = convergence_j_threshold
        self.max_iter = max_iter
        self.n_features = n_features
        self.seed = seed
        self.n_users = 0
        self.n_items = 0
    
    def _sparsify(self, ratings, unobserved_fill_value):
        shape = (self.n_users, self.n_items)
        return_m = np.full(shape, unobserved_fill_value)
        for row in ratings:
            user, item, rating = row
            return_m[int(user), int(item)] = rating
        return return_m
    
    # check converged criteria met
    def _converged(self, i, reached_low_j):
        return i >= self.max_iter or reached_low_j
    
    def fit(self, data): # data in form user_id | item_id | rating
        np.random.seed(self.seed)
        self.n_users, self.n_items = len(np.unique(data[:, 0])), len(np.unique(data[:, 1]))
        self.theta_hats = np.random.rand(self.n_users, self.n_features)
        self.x_hats = np.random.rand(self.n_items, self.n_features)
        self.J_each_iteration = []   
        UNOBSERVED_VALUE = -1
        y = self._sparsify(data, UNOBSERVED_VALUE) # actual ratings matrix
        r = y == UNOBSERVED_VALUE # rated indicator matrix 
        current_iter = 0
        indic_reached_low_j = False
        
        while not self._converged(current_iter, indic_reached_low_j):
            diff = self.theta_hats @ self.x_hats.T - y
            diff[~r] = 0
            J = np.mean(diff**2) / 2
            self.J_each_iteration.append(J)
            if J < self.convergence_j_threshold:
                indic_reached_low_j = True
            new_theta_hats, new_x_hats = self.theta_hats.copy(), self.x_hats.copy()
            for user in range(self.n_users):
                item_js = r[user, :]
                del_J_del_theta_i = np.mean(diff[user, item_js][:, np.newaxis] * self.x_hats[item_js, :], axis = 0)
                new_theta_hats[user] -= alpha * del_J_del_theta_i
            for item in range(self.n_items):
                user_js = r[:, item]
                del_J_del_x_i = np.mean(diff[user_js, item][:, np.newaxis] * self.theta_hats[user_js, :], axis = 0)
                new_x_hats[item] -= alpha * del_J_del_x_i
            self.x_hats = new_x_hats
            self.theta_hats = new_theta_hats
            current_iter += 1
        
        return self
    
    def predict(self, data): # data in form user_id | item_id
        return np.array([self.theta_hats[int(user)].T @ self.x_hats[int(item)] for user, item in data])