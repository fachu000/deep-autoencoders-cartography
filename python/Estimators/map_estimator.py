from abc import abstractmethod


class MapEstimator:
    def __init__(self):
        self.str_name = "Unnamed"

    @abstractmethod
    def estimate_map(self, sampled_map, mask, meta_data):
        """
           :param sampled_map: 3D array of shape n_pts_x by n_pts_y by n_frequencies)
           :param mask: 2D binary array  of shape n_pts_x by n_pts_y with entries equal to 1 if the point is sampled
                        or 0 otherwise. The same mask is used across frequency dimension
           :param meta_data: 2D binary array  of shape n_pts_x by n_pts_y with entries equal to 1 if the point is
                            inside a builing or 0 otherwise. The same meta_data is used across frequency dimension
           :return: 3D array with the same shape as the sampled map
           :
           """

        pass


class BemMapEstimator(MapEstimator):
    def __init__(self, bases_vals=None, **kwargs):
        super().__init__(**kwargs)

        self.bases_vals = bases_vals

    @abstractmethod
    def estimate_bem_coefficient_map(self, sampled_map, mask, meta_data):
        """ Input args as in MapEstimator.estimate_map

        Output: n_pts_x X n_pts_y X num_bases nparray

        """

        pass
