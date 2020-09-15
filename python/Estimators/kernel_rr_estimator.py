import numpy as np
import numpy.matlib
from numpy import linalg as npla
from Estimators.map_estimator import MapEstimator
import matplotlib.pyplot as plt
import pandas as pd


class KernelRidgeRegrEstimator(MapEstimator):

    """
        Arguments:
            x_length:   length of the x_axis  of the area under consideration
            y_length:    length of the y_axis of the area under consideration
            n_grid_points_x : number of grid points along x-axis
            n_grid_points_y : number of grid points along y-axis
            sigma : parameter of the gaussian kernel
            par_lambda: regularization parameter in kernel ridge regression
            estimate_missing_val_only: flag which allows to estimate missing entries only, set to False by default.
        """

    def __init__(self,
                 x_length,
                 y_length,
                 sigma=None,
                 par_lambda=1e-5,
                 estimate_missing_val_only=False):
        super(KernelRidgeRegrEstimator, self).__init__()
        self.str_name = "Kriging"
        self.x_length = x_length
        self.y_length = y_length
        self.sigma = sigma
        self.par_lambda = par_lambda
        self.estimate_missing_val_only = estimate_missing_val_only

    def estimate_map(self, sampled_map,  mask, meta_data):

        """
        :param sampled_map:  the sampled map with incomplete entries, 2D array with shape (n_grid_points_x,
                                                                                           n_grid_points_y)
        :param mask: is a binary array of the same size as the sampled map:
        :return: the reconstructed map,  2D array with the same shape as the sampled map

        """
        # check if the kernel parameter sigma is None, if yes, use the this default value
        mask_comb_meta = mask - meta_data
        if self.sigma is None:
            self.sigma = 5 * np.sqrt(self.x_length * self.y_length / np.sum(mask_comb_meta == 1))

        # obtain kernel_coefficients
        sampled_map = sampled_map[:, :, 0]
        n_grid_points_x = sampled_map.shape[0]
        n_grid_points_y = sampled_map.shape[1]
        avail_measur_indices = np.where(mask_comb_meta == 1)

        x_array = np.linspace(0, self.x_length, n_grid_points_x)
        y_array = np.linspace(0, self.y_length, n_grid_points_y)

        n_measurements = len(avail_measur_indices[0])
        power_meas_vec = np.zeros((n_measurements, 1))
        kernel_matrix = np.zeros((n_measurements, n_measurements))
        all_avail_points = np.array([x_array[avail_measur_indices[1]], y_array[avail_measur_indices[0]]])
        for ind_1 in range(n_measurements):
            power_meas_vec[ind_1] = sampled_map[avail_measur_indices[0][ind_1]][avail_measur_indices[1][ind_1]]
            point_1 = [x_array[avail_measur_indices[1][ind_1]], y_array[avail_measur_indices[0][ind_1]]]
            m_row_inputs_to_kernels = np.matlib.repmat(point_1,  n_measurements, 1).T
            kernel_matrix[ind_1] = self.gaussian_kernel(m_row_inputs_to_kernels, all_avail_points)
        kernel_coeff = np.linalg.inv(kernel_matrix + n_measurements * self.par_lambda * np.eye(n_measurements))\
                                     .dot(power_meas_vec)

        # Estimate the map using obtained kernel coefficients
        estimated_map = np.zeros(sampled_map.shape)
        for ind_y in range(len(y_array)):
            for ind_x in range(len(x_array)):
                if mask_comb_meta[ind_y, ind_x] == 1 and self.estimate_missing_val_only:
                    estimated_map[ind_y, ind_x] = sampled_map[ind_y, ind_x]
                else:
                    query_point = [x_array[ind_x], y_array[ind_y]]
                    query_point_rep = np.matlib.repmat(query_point, n_measurements, 1).T
                    kernel_vec = self.gaussian_kernel(query_point_rep, all_avail_points)
                    estimated_map[ind_y, ind_x] = kernel_vec.dot(kernel_coeff)
        return np.expand_dims(estimated_map, axis=2)

    def gaussian_kernel(self, x1, x2):
        output = np.exp(- npla.norm(np.subtract(x1, x2), axis=0) ** 2 / (self.sigma ** 2))
        return output
