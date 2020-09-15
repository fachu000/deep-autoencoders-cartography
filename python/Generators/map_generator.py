from abc import abstractmethod
from utils.communications import dbm_to_natural, natural_to_db, db_to_natural
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class MapGenerator:
    """ARGUMENTS:

    n_grid_points_x : number of grid points along x-axis
    n_grid_points_y : number of grid points along y-axis

    `

    """

    def __init__(self,
                 x_length=100,
                 y_length=100,
                 n_grid_points_x=32,
                 n_grid_points_y=32,
                 m_basis_functions=np.array([[1]]),
                 noise_power_interval=None
                 ):
        self.x_length = x_length
        self.y_length = y_length
        self.n_grid_points_x = n_grid_points_x
        self.n_grid_points_y = n_grid_points_y
        self.m_basis_functions = m_basis_functions  # num_bases x len(v_sampled_frequencies) matrix
        self.noise_power_interval = noise_power_interval

    def generate(self):
        """
        Returns:
        `map`: Nx x Ny x Nf tensor. map at each freq.
        `meta_map`: Nx x Ny, each entry is 1 if that grid point is inside a building; 0 otherwise.
        """

        num_freqs = self.m_basis_functions.shape[1]
        num_bases = self.m_basis_functions.shape[0]

        num_signal_bases = num_bases - 1 if self.noise_power_interval is not None else num_bases

        # Obtain one power map per basis function
        l_signal_maps, m_meta_map = self.generate_power_map_per_freq(num_signal_bases)

        # Obtain power at each sampled frequency
        t_freq_map = np.zeros(shape=(l_signal_maps[0].shape[0], l_signal_maps[0].shape[1], num_freqs))
        for ind_sampled_freq in range(num_freqs):
            t_freq_map_all_bs = np.zeros(shape=(l_signal_maps[0].shape[0], l_signal_maps[0].shape[1], num_signal_bases))
            for ind_central_freq in range(num_signal_bases):
                t_freq_map_all_bs[:, :, ind_central_freq] = l_signal_maps[ind_central_freq] * self.m_basis_functions[
                    ind_central_freq,
                    ind_sampled_freq]
            t_freq_map[:, :, ind_sampled_freq] = np.sum(t_freq_map_all_bs, axis=2)

        if self.noise_power_interval is not None:
            noise_power_interval_nat = dbm_to_natural(np.array(self.noise_power_interval))

            # add noise to the map
            noise_power = (noise_power_interval_nat[1] - noise_power_interval_nat[0]) * np.random.rand() + \
                          noise_power_interval_nat[0]
            t_freq_map += noise_power

            # add noise map for coefficient visualization
            l_signal_maps.append(noise_power * np.ones((l_signal_maps[0].shape[0], l_signal_maps[0].shape[1])))

        # Output channel power maps
        t_channel_pow = natural_to_db(np.transpose(np.array(l_signal_maps), (1, 2, 0)))

        return natural_to_db(t_freq_map), m_meta_map, t_channel_pow

    # TO BE IMPLEMENTED BY ALL DESCENDANTS
    @abstractmethod
    def generate_power_map_per_freq(self, num_bases):
        """Returns:

        - a list of length num_bases, each one
        with the power map of the corresponding basis function.

        - a meta mask (explain)

        """

        pass

    @staticmethod
    def generate_bases(
            v_central_frequencies=np.array([0]),
            v_sampled_frequencies=np.array([0]),
            fun_base_function=lambda freq: 1,
            b_noise_function=True,
            plot_bases=False):

        """`fun_base_function` is a function of 1 frequency argument. 

        If `b_noise_power==False`, then, this function retuns a
        len(v_central_frequencies) x len(v_sampled_frequencies) matrix
        whose (i,j)-th entry is `fun_base_function(
        v_sampled_frequencies[j] - v_central_freq[i] )`.

        Else, the returned matrix equals the aforementioned matrix
        with an extra row of all 1s.

        """

        num_bases = np.size(v_central_frequencies)
        num_freq = len(v_sampled_frequencies)

        # Generate the basis functions
        bases_vals = np.zeros((num_bases, num_freq))
        for ind_base in range(num_bases):
            for ind_sampled_freq in range(num_freq):
                bases_vals[ind_base, ind_sampled_freq] = fun_base_function(
                    v_sampled_frequencies[ind_sampled_freq] - v_central_frequencies[ind_base])

        # Add the noise function
        if b_noise_function:
            bases_vals = np.vstack((bases_vals, np.ones((1, num_freq))))

        # Normalize bases
        for ind_base in range(bases_vals.shape[0]):
            bases_vals[ind_base, :] = bases_vals[ind_base, :] / sum(bases_vals[ind_base, :])

        # Visualize bases
        if plot_bases:
            num_plot_points = 400
            f_range = np.linspace(v_sampled_frequencies[0], v_sampled_frequencies[-1], num_plot_points)

            v_amplitudes = [0.2, 1] # np.zeros((num_bases + 1, 1))  # db_to_natural(np.array([-97, -90]))

            ammpl_noise_inter = [0.0, 0.02]  # db_to_natural(np.array([-0, -0]))
            noise = 0.05 * np.ones((1, len(f_range))) # (ammpl_noise_inter[1] - ammpl_noise_inter[0]) * np.random.rand(len(f_range)) + \
                    # ammpl_noise_inter[0]

            bases_vals_plot = np.zeros((num_bases, len(f_range)))
            for ind_base in range(num_bases):
                ampl_ind_base = (v_amplitudes[1] - v_amplitudes[
                    0]) * np.random.rand() + v_amplitudes[0]
                for ind_freq in range(len(f_range)):
                    bases_vals_plot[ind_base, ind_freq] = ampl_ind_base * fun_base_function(
                        f_range[ind_freq] - v_central_frequencies[ind_base])
            if b_noise_function:
                bases_vals_plot = np.vstack((bases_vals_plot, noise))

            # Normalize bases for plotting
            # for ind_base in range(bases_vals_plot.shape[0]):
            #     bases_vals_plot[ind_base, :] = bases_vals_plot[ind_base, :] / sum(bases_vals_plot[ind_base, :])

            fig = plt.figure()
            n_curves = bases_vals.shape[0]
            for ind_curv in range(n_curves):
                bases_vals_ind_plot = bases_vals_plot[ind_curv, :]
                label = r'$ \pi_%d(\mathbf{x}) \beta_%d (f)$' % (ind_curv + 1, ind_curv + 1)
                # label = r'$ \pi_{s%d} \beta_%d (f)$' % (ind_curv + 1, ind_curv + 1)
                plt.plot(f_range / 1e6, bases_vals_ind_plot,
                         label=label)
            sum_base = np.sum(bases_vals_plot, axis=0)
            plt.plot(f_range / 1e6, sum_base, linestyle='-', color='m',
                     label=r'$\sum_b \pi_b(\mathbf{x})  \beta_b(f)$'
                     # label = r'$\sum_b \pi_{sb}  \beta_b(f)$'
            )
            plt.legend()
            plt.xlabel('f [MHz]')
            plt.ylabel(r'$ \Psi(\mathbf{x}, f)$')
            # plt.ylabel(r'$ \Upsilon_s(f)$')
            plt.grid()
            plt.show()
            quit()
        return bases_vals

    @staticmethod
    def gaussian_base(x, scale):
        return np.exp(-np.power(x, 2.) / (2 * np.power(scale, 2.)))

    @staticmethod
    def raised_cosine_base(freq, roll_off, bandwidth):
        gamma_l = (1 - roll_off) * bandwidth / 2
        gamma_u = (1 + roll_off) * bandwidth / 2
        if abs(freq) <= gamma_l:
            return 1
        elif gamma_l < abs(freq) <= gamma_u:
            return 1 / 2 * (1 + np.cos(np.pi / (bandwidth * roll_off) * (abs(freq) - gamma_l)))
        else:
            return 1e-13

    @staticmethod
    def ofdm_base(freq, num_carriers, bandwidth):
        pass
