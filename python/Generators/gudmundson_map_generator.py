import tensorflow as tf
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
from scipy.spatial import distance
# import sk_dsp_comm.sigsys as ss
from scipy.stats import multivariate_normal
from Generators.map_generator import MapGenerator
from utils.communications import dbm_to_natural, db_to_natural, natural_to_db


class GudmundsonMapGenerator(MapGenerator):
    """
    Arguments:

    Only one of the following can be set:

        tx_power: num_sources x num_bases matrix with the transmitter power.

        tx_power_interval: length-2 vector. Tx. power of all sources at all bases 
            is chosen uniformly at random in the interval 
            [tx_power_interval[0], tx_power_interval[1]].

        path_loss_exp: path loss exponent of the propagation environment
        eps: small constant avoiding large values of power for small distances
        corr_shad_sigma2 : the variance of the shadowing in dB in correlated shadowing
        corr_base: correlation coefficient , see Gudmundson  model.
        
    Output shape:
        2D tensor with shape:
         (n_grid_points_x, n_grid_points_y)

    """

    def __init__(self,
                 *args,
                 v_central_frequencies=None,
                 tx_power=None,
                 n_tx=2,
                 tx_power_interval=None,
                 path_loss_exp=3,
                 corr_shad_sigma2=10,
                 corr_base=0.95,
                 b_shadowing=False,
                 num_precomputed_shadowing_mats=1,
                 **kwargs
                 ):

        super(GudmundsonMapGenerator, self).__init__(*args, **kwargs)

        assert not(tx_power is not None and tx_power_interval is not None), "tx_power and tx_power_interval cannot be simultaneously set."
        assert v_central_frequencies is not None, "Argument `v_central_frequencies` must be provided."
        self.v_central_frequencies = v_central_frequencies
        self.tx_power = tx_power
        self.n_tx = n_tx
        self.tx_power_interval = tx_power_interval
        self.path_loss_exp = path_loss_exp
        self.corr_shad_sigma2 = corr_shad_sigma2
        self.corr_base = corr_base
        self.b_shadowing = b_shadowing
        self.num_precomputed_shadowing_mats = num_precomputed_shadowing_mats
        if self.b_shadowing:
            self.shadowing_dB = self.generate_shadowing(size=(self.num_precomputed_shadowing_mats,
                                                              self.n_tx))  # buffer to store precomputed maps
            self.ind_shadowing_mat = 0  # next returned map will correspond to self.shadowing_dB[self.ind_shadowing_mat,:,:]
        self.eps = min(self.x_length / self.n_grid_points_x, self.y_length / self.n_grid_points_y)


    def generate_power_map_per_freq(self, num_bases):

        assert len(self.v_central_frequencies) == num_bases
        
        l_maps = []  # Nx x Ny x Nf tensor

        # Convert to natural units
        if self.tx_power_interval:
            tx_power_interval_nat = dbm_to_natural(np.array(self.tx_power_interval))
            n_sources = self.n_tx
        elif self.tx_power.all():
            tx_power_nat = dbm_to_natural(self.tx_power)
            n_sources = self.tx_power.shape[1]
        else:
            tx_power_interval_nat = None
            tx_power_nat = None
            Exception("at least one of tx_power or tx_power_interval must be set.")

        n_bases = self.m_basis_functions.shape[0]

        x_grid, y_grid = self.generate_grid(self.x_length, self.y_length, self.n_grid_points_x,
                                            self.n_grid_points_y)

        source_pos_x = np.min(np.min(x_grid)) + (np.max(np.max(x_grid)) - np.min(np.min(x_grid))) * \
                       np.random.rand(n_sources, 1)
        source_pos_y = np.min(np.min(y_grid)) + (np.max(np.max(y_grid)) - np.min(np.min(y_grid))) * \
                       np.random.rand(n_sources, 1)
        source_pos = np.concatenate((source_pos_x, source_pos_y), axis=1)
        c_light = 3e8

        for freq in self.v_central_frequencies:
            k_val = (c_light / (4 * np.pi * freq)) ** 2
            # generate the pathloss componemt
            path_loss_comp = np.zeros((n_sources, self.n_grid_points_x, self.n_grid_points_y))
            for ind_source in range(n_sources):
                for ind_y in range(self.n_grid_points_y):
                    one_y_rep = y_grid[ind_y]
                    all_p_oney = np.array([x_grid[ind_y], one_y_rep])
                    dist_all_p_oney = npla.norm(np.subtract(source_pos[ind_source, :].reshape(-1, 1), all_p_oney),
                                                axis=0)
                    all_power_oney_all_bases = np.zeros((1, self.n_grid_points_x, n_bases))
                    for ind_base in range(n_bases):
                        if self.tx_power_interval:
                            tx_power_to_use = (tx_power_interval_nat[1] - tx_power_interval_nat[
                                0]) * np.random.rand() + tx_power_interval_nat[0]
                        else:
                            tx_power_to_use = tx_power_nat[ind_base, ind_source]
                        all_power_oney = tx_power_to_use * k_val / (
                                (dist_all_p_oney + self.eps) ** self.path_loss_exp)
                        all_power_oney_all_bases[:, :, ind_base] = all_power_oney
                    path_loss_comp[ind_source, ind_y, :] = np.sum(all_power_oney_all_bases, axis=2)[0]

            # add the shadowing componemt
            if self.b_shadowing:
                # pathloss combined with shadowing
                shadow_map_ind = self.next_shadowing_dB()
                shadowing_reshaped = np.reshape(shadow_map_ind, (n_sources, self.n_grid_points_x, self.n_grid_points_y),
                                                order='F')
                # shadowing_reshaped_rep = np.repeat(shadowing_reshaped[:, :, :,  np.newaxis], self.n_freqs, axis=3)
                map_with_shadowing = natural_to_db(path_loss_comp) + shadowing_reshaped
                generated_map = sum(db_to_natural(map_with_shadowing))
            else:
                generated_map = sum(path_loss_comp)

            l_maps.append(generated_map)  # Nx x Ny matrices

        return l_maps,  np.zeros((self.n_grid_points_x, self.n_grid_points_y)) 


    def next_shadowing_dB(self):
        if not self.ind_shadowing_mat < self.shadowing_dB.shape[0]:
            # No maps left
            self.shadowing_dB = self.generate_shadowing(size=(self.num_precomputed_shadowing_mats,
                                                              self.n_tx))
            self.ind_shadowing_mat = 0
        # There are maps left
        shadowing_map = self.shadowing_dB[self.ind_shadowing_mat, :, :]
        self.ind_shadowing_mat += 1
        return shadowing_map

    def generate_shadowing(self, size=(1, 1)):
        x_grid, y_grid = self.generate_grid(self.x_length, self.y_length, self.n_grid_points_x, self.n_grid_points_y)
        vec_x_grid = x_grid.flatten('F')
        vec_y_grid = y_grid.flatten('F')
        all_points = list(zip(vec_x_grid, vec_y_grid))
        dist_pairs = distance.cdist(np.asarray(all_points), np.asarray(all_points), 'euclidean')
        cov_mat = self.corr_shad_sigma2 * self.corr_base ** dist_pairs
        shadowing_dB = multivariate_normal.rvs(mean=np.zeros((len(vec_x_grid))), cov=cov_mat, size=size)
        return shadowing_dB
      
    def generate_grid(self, x_len, y_len, n_points_x, n_points_y):
        x_grid, y_grid = np.meshgrid(np.linspace(0, x_len, n_points_x),
                                     np.linspace(0, y_len, n_points_y))
        return x_grid, y_grid
