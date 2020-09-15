import numpy as np
import numpy.matlib
from numpy import linalg as npla
from Estimators.map_estimator import MapEstimator


class KNNEstimator(MapEstimator):

    def __init__(self,
                 x_length,
                 y_length,
                 k_neigh=5):
        super(KNNEstimator, self).__init__()
        self.str_name = "KNN"
        self.x_length = x_length
        self.y_length = y_length
        self.k_neigh = k_neigh

    def estimate_map(self, sampled_map, mask, meta_data):
        """
        :param sampled_map:  is the 3D array that contains the samples
        :param mask: is a binary array of the same size as the sampled map
        :param neighbours_1d_one_side: number of neighbours along one dimension, for 2D arrays, the number of neighbours is
        neighbours_1d*neighbours_1d
        :return: the reconstructed map,  2D array with the same shape as the sampled map
        """
        sampled_map = sampled_map[:, :, 0]
        n_points_x = sampled_map.shape[0]
        n_points_y = sampled_map.shape[1]
        estimated_map = np.zeros(sampled_map.shape)
        mask_comb_meta = mask - meta_data
        short_circuit_here = False
        for ind_y in range(n_points_y):
            for ind_x in range(n_points_x):
                if mask_comb_meta[ind_y, ind_x] == 1 and not short_circuit_here:
                    estimated_map[ind_y, ind_x] = sampled_map[ind_y, ind_x]
                else:
                    avail_measur_indices = np.where(mask_comb_meta == 1)
                    n_measurements = len(avail_measur_indices[0])
                    x_array = np.linspace(0, self.x_length, n_points_x)
                    y_array = np.linspace(0, self.y_length, n_points_y)
                    current_point = [x_array[ind_x], y_array[ind_y]]
                    all_avail_points = np.array(
                        [x_array[avail_measur_indices[1]], y_array[avail_measur_indices[0]]])
                    current_point_rep = np.matlib.repmat(current_point, n_measurements, 1).T
                    dist_to_all_measur = npla.norm(np.subtract(current_point_rep, all_avail_points), axis=0)
                    dist_to_all_measur_sorted = np.sort(dist_to_all_measur)[0:self.k_neigh]
                    neighbours = []
                    for val in dist_to_all_measur_sorted:
                        ind_neig = np.where(dist_to_all_measur == val)
                        neighbours_ind = [avail_measur_indices[0][ind_neig], avail_measur_indices[1][ind_neig]]
                        neighbours = np.append(neighbours, sampled_map[tuple(neighbours_ind)])
                    estimated_map[ind_y, ind_x] = np.mean(neighbours)

        return np.expand_dims(estimated_map, axis=2)
