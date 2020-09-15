import numpy as np


class MapSampler:
    """
        
    ARGUMENTS:

        `sampling_factor`: can be:

              - fraction between 0 and 1 determining on average the percentage of entries to select
                as samples, i.e  sampling_factor= 0.3 will allow selection of 30 % of total entries of 
                the map that are different from np.nan. 

              - Interval (list or tuple of length 2). The aforementioned fraction is drawn uniformly at random 
                within the interval [sampling_factor[0], sampling_factor[1] ] each time a map is sampled.
            
        `std_noise`: standard deviation of the Gaussian noise in the  sampled map
        `sample_threshold`: set to -200 dBW  ( this is the rx value returned by wireless Insite when receivers are out
        of the coverage are)

        `set_unobserved_val_to_minus1`: flag that is used to set the unsampled entries to -1
        """

    def __init__(self,
                 v_sampling_factor=[],
                 std_noise=0,
                 ):
        self.v_sampling_factor = v_sampling_factor
        self.std_noise = std_noise
        self.set_unobserved_val_to_minus1 = False

    def sample_map(self, t_map_to_sample, m_meta_map):

        """
        Returns:
        `t_sampled map`: Nx x Ny x Nf tensor
        `m_mask`: Nx x Ny, each entry is 1 if that grid point is sampled; 0 otherwise, same mask is applied along Nf dimension.
        """
        if np.size(self.v_sampling_factor) == 1:
            sampling_factor = self.v_sampling_factor
        elif np.size(self.v_sampling_factor) == 2:
            sampling_factor = np.round_((self.v_sampling_factor[1] - self.v_sampling_factor[0]) * np.random.rand() +
                                        self.v_sampling_factor[0], decimals=2)
        else:
            Exception("invalid value of v_sampling_factor")
        shape_in = t_map_to_sample.shape
        m_mask = np.ones((shape_in[0], shape_in[1]))
        if sampling_factor == 1:
            sampled_map = t_map_to_sample
            m_mask_ret = m_mask
        else:
            m_map_to_sample = np.reshape(t_map_to_sample, (shape_in[0] * shape_in[1], shape_in[2]),
                                           order='F')
            v_mask = m_mask.flatten('F')
            v_meta_data = m_meta_map.flatten('F')
            unrelevant_ind = np.where(v_meta_data == 1)[0]
            indices_to_sampled_from = np.where(v_meta_data == 0)[0]
            unobs_val_ind = np.random.choice(indices_to_sampled_from,
                                             size=int((1 - sampling_factor) * len(indices_to_sampled_from)),
                                             replace=False)
            all_unobs_ind_in = list(map(int, np.concatenate((unrelevant_ind, unobs_val_ind),
                                                            axis=0)))
            if self.set_unobserved_val_to_minus1:
                m_map_to_sample[all_unobs_ind_in, :] = -1
            else:
                m_map_to_sample[all_unobs_ind_in, :] = 0
            v_mask[list(map(int, all_unobs_ind_in))] = 0
            m_mask_ret = np.reshape(v_mask, (shape_in[0], shape_in[1]), order='F')
            sampled_map = np.reshape(m_map_to_sample, t_map_to_sample.shape, order='F')
        t_sampled_map_ret = sampled_map + np.multiply(
            np.random.normal(loc=0, scale=self.std_noise, size=t_map_to_sample.shape),
            np.expand_dims(m_mask_ret, axis=2))
        return t_sampled_map_ret, m_mask_ret

    def resample_map(self, t_sampled_map, m_mask, v_split_frac):
        """
                Returns:
                `t_sampled map_in`: Nx x Ny x Nf tensor
                `m_mask_in`: Nx x Ny, indicates which entries in sampled map_in are sampled, each entry is 1 if that grid
                 point is sampled; 0 otherwise, same mask is applied along Nf dimension.
                `t_sampled map_out`: Nx x Ny x Nf tensor
                `m_mask_out`: Nx x Ny, indicates which entries in sampled map_out are sampled, each entry is 1 if that
                grid point is sampled; 0 otherwise, same mask is applied along Nf dimension.
                """
        shape_in = t_sampled_map.shape
        v_mask_in = m_mask.flatten('F')
        v_mask_target = m_mask.flatten('F')
        m_map_to_resample_in = np.reshape(t_sampled_map, (shape_in[0] * shape_in[1], shape_in[2]),
                                            order='F')
        m_map_to_resample_out = np.reshape(t_sampled_map, (shape_in[0] * shape_in[1], shape_in[2]),
                                            order='F')
        indices_to_sampled_from = np.where(v_mask_in == 1)[0]
        unobs_val_ind_in = np.random.choice(indices_to_sampled_from,
                                            size=int((1 - v_split_frac[0]) * len(indices_to_sampled_from)),
                                            replace=False)
        unobs_val_ind_target = np.random.choice(indices_to_sampled_from,
                                                size=int((1 - v_split_frac[1]) * len(indices_to_sampled_from)),
                                                replace=False)

        if self.set_unobserved_val_to_minus1:
            m_map_to_resample_in[unobs_val_ind_in, :] = -1
            m_map_to_resample_out[unobs_val_ind_target, :] = -1
        else:
            m_map_to_resample_in[unobs_val_ind_in, :] = 0
            m_map_to_resample_out[unobs_val_ind_target, :] = 0
        v_mask_in[list(map(int, unobs_val_ind_in))] = 0
        v_mask_target[list(map(int, unobs_val_ind_target))] = 0
        t_resampled_map_in = np.reshape(m_map_to_resample_in, t_sampled_map.shape, order='F')
        t_resampled_map_out = np.reshape(m_map_to_resample_out, t_sampled_map.shape, order='F')
        m_mask_in_ret = np.reshape(v_mask_in, (shape_in[0], shape_in[1]), order='F')
        m_mask_target_ret = np.reshape(v_mask_target, (shape_in[0], shape_in[1]), order='F')
        t_resampled_map_in_ret = t_resampled_map_in + np.multiply(
            np.random.normal(loc=0, scale=self.std_noise, size=t_sampled_map.shape),
            np.expand_dims(m_mask_in_ret, axis=2))
        t_resampled_map_out_ret = t_resampled_map_out + np.multiply(
            np.random.normal(loc=0, scale=self.std_noise, size=t_sampled_map.shape),
            np.expand_dims(m_mask_target_ret, axis=2))
        return t_resampled_map_in_ret, m_mask_in_ret, t_resampled_map_out_ret, m_mask_target_ret


def list_complement_elements(list_1, list_2):
    complement_list = []
    for num in list_1:
        if num not in list_2:
            complement_list.append(num)
    return np.array(complement_list)
