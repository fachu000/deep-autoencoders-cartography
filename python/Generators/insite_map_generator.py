from Generators.map_generator import MapGenerator
from utils.communications import dbm_to_natural, natural_to_dbm, dbm_to_db, db_to_natural
import pandas as pd
import numpy as np
import cv2

building_threshold = -200  # Threshold in dBm to determine building locations


class InsiteMapGenerator(MapGenerator):
    def __init__(
            self,
            num_tx_per_channel=2,
            l_file_num=np.arange(1, 40),
            large_map_size=244,
            # closest square to the map size provided by Wireless Insight soft is 59536 = 244^2
            filter_map=True,
            filter_size=3,
            inter_grid_points_dist_factor=1,  # set to an integer greater than 1
            *args,
            **kwargs):

        super(InsiteMapGenerator, self).__init__(*args, **kwargs)
        self.num_tx_per_channel = num_tx_per_channel
        self.l_file_num = l_file_num
        self.large_map_size = large_map_size
        self.filter_map = filter_map
        self.filter_size = filter_size
        self.inter_grid_points_dist_factor = inter_grid_points_dist_factor

    def generate_power_map_per_freq(self, num_bases):

        l_maps = []

        if self.l_file_num[0] == 50 and self.l_file_num[-1] == 51 and self.num_tx_per_channel == 2:
            # reconstructing a Wireless Insite map taken with a higher resolution (used in the conference paper)
            rx_power_tx1 = np.array(
                pd.read_csv("Generators/remcom_maps/power_tx50.txt",
                            delim_whitespace=True,
                            skipinitialspace=True))
            rx_power_tx2 = np.array(
                pd.read_csv("Generators/remcom_maps/power_tx51.txt",
                            delim_whitespace=True,
                            skipinitialspace=True))
            rx_power_tx1_dBW = dbm_to_db(np.reshape(rx_power_tx1,
                                                    (self.n_grid_points_x, self.n_grid_points_y), order='C'))
            rx_power_tx2_dBW = dbm_to_db(np.reshape(rx_power_tx2,
                                                    (self.n_grid_points_x, self.n_grid_points_y), order='C'))
            rx_pow_tot_dBw = db_to_natural(rx_power_tx1_dBW) + db_to_natural(rx_power_tx2_dBW)

            l_maps.append(rx_pow_tot_dBw)
            
        else:

            # Generate coordinates of random patch
            patch_indices = np.random.choice(self.large_map_size -
                                             self.n_grid_points_x * self.inter_grid_points_dist_factor,
                                             size=2)

            for basis_ind in range(num_bases):
                map_this_frequency = np.zeros(
                    (self.n_grid_points_x, self.n_grid_points_y))
                assert len(self.l_file_num) >= self.num_tx_per_channel, 'The number of map extraction files should be ' \
                                                                        'greater or equal to the number of transmitters per channel'
                files_ind = np.random.choice(self.l_file_num,
                                             size=self.num_tx_per_channel,
                                             replace=False)
                for ind_tx in range(self.num_tx_per_channel):
                    # Choose a file and get the large map
                    file_name = 'power_tx%s.p2m' % files_ind[ind_tx]
                    large_map_tx = np.array(
                        pd.read_csv(
                            'Generators/remcom_maps/'
                            + file_name,
                            delim_whitespace=True,
                            skiprows=[0],
                            usecols=['Power(dBm)']))
                    large_map_tx_resh = dbm_to_natural(np.reshape(large_map_tx[0:(self.large_map_size ** 2)],
                                                                  (self.large_map_size, self.large_map_size),
                                                                  order='C'))
                    # Extract patch from the file
                    maps_as_patch = self.get_patch(large_map_tx_resh,
                                                   patch_indices)

                    map_this_frequency += maps_as_patch

                # Filter the map
                if self.filter_map:
                    filter_to_use = np.ones(
                        (self.filter_size, self.filter_size),
                        np.float32) / (self.filter_size * self.filter_size)
                    map_this_frequency_filter = cv2.filter2D(map_this_frequency, -1,
                                                             filter_to_use)
                else:
                    map_this_frequency_filter = map_this_frequency

                l_maps.append(map_this_frequency_filter)  # list of Nx x Ny matrices

        return l_maps, obtain_meta_map(l_maps[0])

    def get_patch(self, large_image, startRow_and_Col):
        if self.inter_grid_points_dist_factor > 1:
            v_patch_indices_y = np.array(range(startRow_and_Col[0], startRow_and_Col[0] +
                                               self.inter_grid_points_dist_factor * self.n_grid_points_y))
            v_patch_indices_x = np.array(range(startRow_and_Col[1], startRow_and_Col[1] +
                                               self.inter_grid_points_dist_factor * self.n_grid_points_x))
            v_coarse_patch_indices_y = v_patch_indices_y[0::self.inter_grid_points_dist_factor]
            v_coarse_patch_indices_x = v_patch_indices_x[0::self.inter_grid_points_dist_factor]
            return large_image[v_coarse_patch_indices_y.reshape(-1, 1), v_coarse_patch_indices_x.reshape(1, -1)]
        else:
            return large_image[startRow_and_Col[0]:startRow_and_Col[0] +
                                                   self.n_grid_points_y,
                   startRow_and_Col[1]:startRow_and_Col[1] +
                                       self.n_grid_points_x]


def obtain_meta_map(m_map):
    """
    Returns:
        `m_meta_map_ret`: Nx x Ny matrix where each entry is 1 if that grid point is inside the building,
         0 otherwise.
    """
    m_meta_map = np.zeros((m_map.shape[0], m_map.shape[1]))
    v_meta_map = m_meta_map.flatten('F')
    v_map = m_map.flatten('F')
    ind_pts_in_building = np.where(
        v_map < dbm_to_natural(building_threshold))[0]
    v_meta_map[list(map(int, ind_pts_in_building))] = 1
    m_meta_map_ret = np.reshape(v_meta_map, m_meta_map.shape, order='F')
    return m_meta_map_ret
