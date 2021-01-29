from cvxpy import *
import numpy as np
import numpy.matlib
from numpy import linalg as npla
from Estimators.map_estimator import MapEstimator
import matplotlib.pyplot as plt
import pandas as pd


class GroupLassoMKEstimator(MapEstimator):
    """
        Arguments:
            x_length:   length of the x_axis  of the area under consideration
            y_length:    length of the y_axis of the area under consideration
            sigma : parameter of the gaussian kernel
            par_lambda: regularization parameter in the multi-kernel
            estimate_missing_val_only: flag which allows to estimate missing entries only, set to False by default.
        """

    def __init__(self,
                 x_length,
                 y_length,
                 sigma=None,
                 str_name=None,
                 n_kernels=20,
                 par_lambda=5e-5,
                 use_laplac_kernel=True,
                 estimate_missing_val_only=False):
        super(GroupLassoMKEstimator, self).__init__()
        self.str_name = str_name
        self.x_length = x_length
        self.y_length = y_length
        self.sigma = sigma
        self.n_kernels = n_kernels
        self.par_lambda = par_lambda
        self.use_laplac_kernel = use_laplac_kernel
        self.estimate_missing_val_only = estimate_missing_val_only

    def estimate_map(self, sampled_map, mask, meta_data):

        """
        :param sampled_map:  the sampled map with incomplete entries, 2D array with shape (n_grid_points_x,
                                                                                           n_grid_points_y)
        :param mask: is a binary array of the same size as the sampled map:
        :return: the reconstructed map,  2D array with the same shape as the sampled map

        """
        mask_comb_meta = mask - meta_data
        # check if the kernel parameter sigma is None, if yes, use the this default value
        if self.sigma is None:
            self.sigma = 3 * np.sqrt(self.x_length * self.y_length / np.sum(mask_comb_meta == 1))

        # sigmas = np.random.uniform(size=self.n_kernels) * self.sigma
        if self.use_laplac_kernel:
            sigmas = np.linspace(0.1, 1, self.n_kernels) * self.sigma
        else:
            sigmas = np.linspace(0.005, 1, self.n_kernels) * (self.x_length)

        # obtain kernel_coefficients
        sampled_map = sampled_map[:, :, 0]
        n_grid_points_x = sampled_map.shape[0]
        n_grid_points_y = sampled_map.shape[1]
        avail_measur_indices = np.where(mask_comb_meta == 1)

        x_array = np.linspace(0, self.x_length, n_grid_points_x)
        y_array = np.linspace(0, self.y_length, n_grid_points_y)

        n_measurements = len(avail_measur_indices[0])
        power_meas_vec = np.zeros(n_measurements)

        kernel_matrices = np.zeros((n_measurements, n_measurements, self.n_kernels))
        all_avail_points = np.array([x_array[avail_measur_indices[1]], y_array[avail_measur_indices[0]]])
        for ind_1 in range(n_measurements):
            power_meas_vec[ind_1] = sampled_map[avail_measur_indices[0][ind_1]][avail_measur_indices[1][ind_1]]
            point_1 = [x_array[avail_measur_indices[1][ind_1]], y_array[avail_measur_indices[0][ind_1]]]
            m_row_inputs_to_kernels = np.matlib.repmat(point_1, n_measurements, 1).T
            for ind_kern in range(self.n_kernels):
                kernel_matrices[ind_1, :, ind_kern] = self.kernel_function(m_row_inputs_to_kernels, all_avail_points,
                                                                            sigmas[ind_kern])

        kernel_coeffs = Variable(self.n_kernels * n_measurements)
        all_kernel_matrices = np.reshape(kernel_matrices, (n_measurements, n_measurements * self.n_kernels), order='F')
        chol_matrices = np.zeros(kernel_matrices.shape)
        for ind_kern in range(self.n_kernels):
            chol_matrices[:, :, ind_kern] = np.linalg.cholesky(
                kernel_matrices[:, :, ind_kern] + 1e-2 * np.eye(n_measurements))  # or 1e-7 worked better with laplacian

        all_chol_matrices = np.reshape(chol_matrices, (n_measurements, n_measurements * self.n_kernels), order='F')
        gamma = Parameter(nonneg=True)

        objective = Minimize(0.5 * sum_squares((all_kernel_matrices @ kernel_coeffs) - power_meas_vec) +
                             gamma * (norm(all_chol_matrices @ kernel_coeffs, 2)))
        p = Problem(objective)
        gamma_value = self.par_lambda
        gamma.value = gamma_value
        result = p.solve(solver='ECOS')
        kernel_coeffs = kernel_coeffs.value
        # print(kernel_coeffs)
        
        # Estimate the map using obtained kernel coefficients
        estimated_map = np.zeros(sampled_map.shape)
        for ind_y in range(len(y_array)):
            for ind_x in range(len(x_array)):
                if mask_comb_meta[ind_y, ind_x] == 1 and self.estimate_missing_val_only:
                    estimated_map[ind_y, ind_x] = sampled_map[ind_y, ind_x]
                else:
                    query_point = [x_array[ind_x], y_array[ind_y]]
                    query_point_rep = np.matlib.repmat(query_point, n_measurements, 1).T
                    kernel_vecs = np.zeros((n_measurements, self.n_kernels))
                    for ind_kern in range(self.n_kernels):
                        kernel_vecs[:, ind_kern] = self.kernel_function(query_point_rep, all_avail_points,
                                                                         sigmas[ind_kern])
                    estimated_map[ind_y, ind_x] = (kernel_vecs.flatten('F').dot(kernel_coeffs))
        return np.expand_dims(estimated_map, axis=2)

    def kernel_function(self, x1, x2, sigma):
        if self.use_laplac_kernel:
            exponent = 1
        else:
            # use the universal kernel: RBF
            exponent = 2
        output = np.exp(- npla.norm(np.subtract(x1, x2), axis=0) ** exponent / (sigma ** 2))
        return output
