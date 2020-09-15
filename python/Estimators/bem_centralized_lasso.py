from cvxpy import *
import numpy as np
import numpy.matlib
from numpy import linalg as npla
from utils.communications import db_to_natural, natural_to_db
from Estimators.map_estimator import BemMapEstimator

eps_log = 1e-20  # Avoid computation of log of negative numbers


class BemCentralizedLassoKEstimator(BemMapEstimator):
    """
    Implementation of the centralized algorithm in "Distributed Spectrum Sensing for Cognitive Radio
      Networks by Exploiting Sparsity" by Juan Andrés Bazerque and Georgios B. Giannakis.
    
        Arguments:
            x_length:   length of the x_axis  of the area under consideration
            y_length:    length of the y_axis of the area under consideration
            n_grid_points_x : number of grid points along x-axis
            n_grid_points_y : number of grid points along y-axis
            sigma : parameter of the gaussian kernel
            regular_par: regularization parameter in the multi-kernel
            estimate_missing_val_only: flag which allows to estimate missing entries only, 
               set to False by default. When True, the measurements are returned as
               the map estimate at the measurement locations.

            `estimate_noise_psd`: if False, it is assumed that \sigma_r**2 = 0 for all r.
        """

    def __init__(self,
                 x_length=1,
                 y_length=1,
                 regular_par=6e-14,
                 n_grid_points_x=8,
                 n_grid_points_y=8,
                 n_canditate_points_x=10,
                 n_canditate_points_y=10,
                 path_loss_exp=3,
                 estimate_missing_val_only=False,
                 estimate_noise_psd=True,
                 **kwargs):
        super(BemCentralizedLassoKEstimator, self).__init__(**kwargs)
        self.str_name = "Bazerque and Giannakis"
        self.x_length = x_length
        self.y_length = y_length
        self.regular_par = regular_par
        self.n_grid_points_x = n_grid_points_x
        self.n_grid_points_y = n_grid_points_y
        self.n_canditate_points_x = n_canditate_points_x
        self.n_canditate_points_y = n_canditate_points_y
        self.x_array = np.linspace(0, self.x_length, self.n_grid_points_x)
        self.y_array = np.linspace(0, self.y_length, self.n_grid_points_y)
        self.x_candi_array = np.linspace(0, self.x_length, self.n_canditate_points_x)
        self.y_candi_array = np.linspace(0, self.y_length, self.n_canditate_points_y)
        self.path_loss_exp = path_loss_exp
        self.small_const = np.minimum(self.x_length / self.n_grid_points_x, self.y_length / self.n_grid_points_y)
        self.estimate_missing_val_only = estimate_missing_val_only
        self.estimate_noise_psd = estimate_noise_psd

    def estimate_map(self, sampled_map, mask, meta_data):

        mask_comb_meta = mask - meta_data

        # number of frequencies and bases
        num_freqs = sampled_map.shape[2]

        # obtain coefficients
        expansion_coeffs, avg_noise_estimate = self.estimate_coeffients(sampled_map, mask)

        # Evaluate map at the query grid
        t_estimated_map = np.full((self.n_grid_points_y, self.n_grid_points_x, num_freqs), fill_value=None, dtype=float)
        for ind_y in range(len(self.y_array)):
            for ind_x in range(len(self.x_array)):
                if mask_comb_meta[ind_y, ind_x] == 1 and self.estimate_missing_val_only:
                    t_estimated_map[ind_y, ind_x, :] = sampled_map[ind_y, ind_x, :]
                else:
                    m_psd_basis = self.psd_basis(self.x_array[ind_x], self.y_array[ind_y])
                    v_estimated_psd = m_psd_basis @ expansion_coeffs + avg_noise_estimate
                    t_estimated_map[ind_y, ind_x, :] = natural_to_db(
                        np.maximum(v_estimated_psd[:, 0], eps_log * np.ones((1, num_freqs))))

        return t_estimated_map

    def psd_basis(self, x_coord, y_coord):
        """This function returns B_r on the paper when `x_coord` and
        `y_coord` correspond to the r-th measurement location.

        Returns:

        - `m_psd_basis`: num_freqs x num_coefficients matrix such that the psd at point (x_coord, y_coord) equals
        `m_psd_basis @ v_coefficients + noise_psd'.

        """

        num_bases = self.bases_vals.shape[0]
        num_freqs = self.bases_vals.shape[1]

        gains_to_all_points = self.gain_basis(x_coord, y_coord)
        m_psd_basis = np.full((num_freqs, num_bases * self.n_canditate_points_x * self.n_canditate_points_y),
                              fill_value=None, dtype=float)
        for ind_fr in range(num_freqs):
            all_gains_all_bases = np.zeros((num_bases,
                                            self.n_canditate_points_y, self.n_canditate_points_x))
            for ind_base in range(num_bases):
                all_gains_all_bases[ind_base, :, :] = gains_to_all_points * self.bases_vals[ind_base, ind_fr]
            m_psd_basis[ind_fr, :] = np.ndarray.flatten(all_gains_all_bases, order='F')

        return m_psd_basis

    def gain_basis(self, x_coord, y_coord):
        """
        Used to compute the \gamma_sr 's in the paper. 
        """
        current_point = np.array([x_coord, y_coord]).reshape(-1, 1)
        query_gains_to_all_points = np.zeros((self.n_canditate_points_y, self.n_canditate_points_x))
        for ind_query_y in range(self.n_canditate_points_y):
            one_qy_rep = np.matlib.repmat(self.y_candi_array[ind_query_y], self.n_canditate_points_x, 1).T
            all_p_oney = np.array([self.x_candi_array, one_qy_rep[0]])
            dist_all_p_oneyq = npla.norm(np.subtract(current_point, all_p_oney), axis=0)
            query_gains_to_all_points[ind_query_y, :] = 1 / (
                    (dist_all_p_oneyq + self.small_const) ** self.path_loss_exp)
        return query_gains_to_all_points

    def estimate_coeffients(self, sampled_map, mask):

        # number of frequencies and bases
        num_freqs = sampled_map.shape[2]
        num_bases = self.bases_vals.shape[0]

        num_coeffs = num_bases * self.n_canditate_points_x * self.n_canditate_points_y

        avail_measur_indices = np.where(mask == 1)
        all_avail_points = np.array([self.x_array[avail_measur_indices[1]], self.y_array[avail_measur_indices[0]]])
        n_measurements = len(avail_measur_indices[0])

        # Obtain psd measurements
        m_psd_meas = np.zeros((n_measurements, num_freqs, 1))
        t_br = np.full((n_measurements, num_freqs, num_coeffs), fill_value=None, dtype=float)

        for ind_meas in range(n_measurements):
            t_br[ind_meas, :, :] = self.psd_basis(all_avail_points[0, ind_meas], all_avail_points[1, ind_meas])
            m_psd_meas[ind_meas, :, 0] = db_to_natural(sampled_map[avail_measur_indices[0][ind_meas],
                                                       avail_measur_indices[1][ind_meas], :])

        v_psd_meas = np.reshape(m_psd_meas, (m_psd_meas.shape[0] * m_psd_meas.shape[1], 1), order='F')
        m_br = np.reshape(t_br, (n_measurements * num_freqs, num_coeffs), order='F')

        # CVX
        expansion_coeffs = Variable(
            (num_coeffs, 1))
        n_noise_vrs = 1  # n_measurements * num_freqs
        noise_power = Variable((n_noise_vrs, 1))

        reg_par = Parameter(nonneg=True)

        # loop_constraints = []
        # for ind_fr in range(num_freqs):
        #     if ind_fr == 0:
        #         pass
        #     loop_constraints = loop_constraints + \
        #         [noise_power[ind_fr * n_measurements: n_measurements * (ind_fr + 1),
        #                                             0] == noise_power[0:n_measurements, 0]]

        # loop_constraints = []
        # for ind_sig in range(n_measurements * num_freqs):
        #     if ind_sig == 0:
        #         pass
        #     loop_constraints = loop_constraints + \
        #                        [noise_power[ind_sig, 0] == noise_power[0, 0]]

        constraints = [expansion_coeffs >= np.zeros((num_coeffs, 1))
                       , noise_power >= np.zeros((n_noise_vrs, 1))
                       ] # + loop_constraints

        fitting_term = sum_squares(v_psd_meas - m_br @ expansion_coeffs - noise_power)

        objective = Minimize(fitting_term + reg_par * norm1(expansion_coeffs))

        p = Problem(objective, constraints=constraints)
        reg_par.value = self.regular_par
        result = p.solve(verbose=True, rho=5e-9, sigma=1e-10)  #
        expansion_coeffs = expansion_coeffs.value
        print('The minimum coefficient value is %f' % np.amin(expansion_coeffs))
        avg_noise_estimate = np.mean(noise_power.value)
        return expansion_coeffs, avg_noise_estimate

    def estimate_bem_coefficient_map(self, sampled_map, mask, meta_data):

        # number of bases
        num_bases = self.bases_vals.shape[0]

        # obtain coefficients
        expansion_coeffs, avg_noise_estimate = self.estimate_coeffients(sampled_map, mask)

        coeff_resh = np.reshape(expansion_coeffs,
                                (num_bases, self.n_canditate_points_y * self.n_canditate_points_x),
                                order='F')
        estimated_bem_coeffs = np.zeros((self.n_grid_points_y,
                                         self.n_grid_points_x,
                                         num_bases))
        # Compute the expansion coefficients at the query grid
        for ind_y in range(len(self.y_array)):
            for ind_x in range(len(self.x_array)):
                query_gains_to_all_points = self.gain_basis(self.x_array[ind_x], self.y_array[ind_y])
                v_bems_coeff = np.maximum(coeff_resh.dot(np.ndarray.flatten(query_gains_to_all_points, order='F')),
                                          eps_log * np.ones((1, num_bases)))

                estimated_bem_coeffs[ind_y, ind_x, :] = natural_to_db(v_bems_coeff)
        estimated_bem_coeffs_and_noise = np.dstack(
            (estimated_bem_coeffs,
             natural_to_db(np.absolute(avg_noise_estimate) * np.ones((self.n_grid_points_y, self.n_grid_points_x)))))
        return estimated_bem_coeffs_and_noise
