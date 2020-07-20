import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters, max_iter, max_convergence_change_cnt, random_seed):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.max_convergence_change_cnt = max_convergence_change_cnt
        self.random_seed = random_seed

    def fit(self, X):
        random.seed(self.random_seed)

        n_rows = X.shape[0]
        self.grps = np.zeros(n_rows)
        # set initial clusters
        self.grps[random.sample(range(0, n_rows - 1), self.n_clusters)] = np.arange(self.n_clusters)
        self.changed_cnt_each_iteration = []

        for i in range(self.max_iter):
            # calculate centroids (use mean for now)
            self.centroids = np.array([np.mean(X[self.grps == grp], axis = 0) for grp in range(self.n_clusters)])

            # calculate and reassign to nearest cluster (use Euclidean distance for now)
            new_grps = np.array([np.argmin(np.sum((row - self.centroids)**2, axis = 1)**0.5) for row in X])
            changed_cnt = sum(new_grps != self.grps)
            self.changed_cnt_each_iteration.append(changed_cnt)
            self.grps = new_grps
            self.n_iter = i
            if changed_cnt == 0:
                break

        self.inertia_ = np.sum(np.sum((X - self.centroids[self.grps])**2, axis = 1))
        return self

    def predict(self, X):
        new_grps = np.array([np.argmin(np.sum((row - self.centroids)**2, axis = 1)**0.5) for row in X])
        return new_grps


if __name__ == '__main__':
    pass