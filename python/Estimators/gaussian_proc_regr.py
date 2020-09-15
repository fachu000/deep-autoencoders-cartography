
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from Estimators.map_estimator import MapEstimator
# import matplotlib.pyplot as plt
# import pandas as pd


class GaussianProcessRegrEstimator(MapEstimator):
    """
        Arguments:
            x_length:   length of the x_axis  of the area under consideration
            y_length:    length of the y_axis of the area under consideration
            estimate_missing_val_only: flag which allows to estimate missing entries only, set to False by default.
        """

    def __init__(self,
                 x_length,
                 y_length,
                 estimate_missing_val_only=False):
        super(GaussianProcessRegrEstimator, self).__init__()
        self.str_name = "GPR"
        self.x_length = x_length
        self.y_length = y_length
        self.estimate_missing_val_only = estimate_missing_val_only

    def estimate_map(self, sampled_map, mask, meta_data):

        """
        :param sampled_map:  the sampled map with incomplete entries, 2D array with shape (n_grid_points_x,
                                                                                           n_grid_points_y)
        :param mask: is a binary array of the same size as the sampled map:
        :param meta_map: is a binary array of the same size as the sampled map
        :return: the reconstructed map,  2D array with the same shape as the sampled map

        """
        mask_comb_meta = mask - meta_data

        # obtain kernel_coefficients
        sampled_map = sampled_map[:, :, 0]
        n_grid_points_x = sampled_map.shape[0]
        n_grid_points_y = sampled_map.shape[1]
        avail_measur_indices = np.where(mask_comb_meta == 1)

        x_array = np.linspace(0, self.x_length, n_grid_points_x)
        y_array = np.linspace(0, self.y_length, n_grid_points_y)

        n_measurements = len(avail_measur_indices[0])
        power_meas_vec = np.zeros(n_measurements)

        all_avail_points = np.array([x_array[avail_measur_indices[1]], y_array[avail_measur_indices[0]]])
        all_avail_points_pro = np.transpose(all_avail_points)

        for ind_1 in range(n_measurements):
            power_meas_vec[ind_1] = sampled_map[avail_measur_indices[0][ind_1]][avail_measur_indices[1][ind_1]]

        # Fit the data
        kernel = RBF()
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       random_state=1,
                                       alpha=3e-1,
                                       n_restarts_optimizer=2,
                                       normalize_y=True).fit(all_avail_points_pro, power_meas_vec)
        gpr.score(all_avail_points_pro, power_meas_vec)

        # Estimate the map
        estimated_map = np.zeros(sampled_map.shape)
        for ind_y in range(len(y_array)):
            for ind_x in range(len(x_array)):
                if mask_comb_meta[ind_y, ind_x] == 1 and self.estimate_missing_val_only:
                    estimated_map[ind_y, ind_x] = sampled_map[ind_y, ind_x]
                else:
                    query_point = np.array([x_array[ind_x], y_array[ind_y]]).reshape(1, -1)
                    estimated_map[ind_y, ind_x] = gpr.predict(query_point, return_std=False)
        return np.expand_dims(estimated_map, axis=2)

