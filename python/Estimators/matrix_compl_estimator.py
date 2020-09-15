import numpy as np
from cvxpy import *
from Estimators.map_estimator import MapEstimator


class MatrixComplEstimator(MapEstimator):

    def __init__(self,
                 tau):
        self.str_name = "Nuclear norm min."
        self.tau = tau

    def estimate_map(self, sampled_map, mask, meta_data):
        sampled_map = sampled_map[:, :, 0]
        mean_v = sampled_map.mean()
        sampled_map = sampled_map - mean_v * mask
        X = Variable(shape=sampled_map.shape)
        objective = Minimize(self.tau * norm(X, "nuc") +
                             0.5 * sum_squares(multiply(mask, X - sampled_map)))
        problem = Problem(objective)
        problem.solve(solver=CVXOPT, verbose=False)
        # print(X.value + mean_v)
        return np.expand_dims((X.value + mean_v), axis=2)

