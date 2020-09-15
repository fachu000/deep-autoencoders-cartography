import numpy as np
from numpy import linalg as npla
from joblib import Parallel, delayed
import multiprocessing


class Simulator:
    """
       Arguments:
              n_runs: the number of monte carlo runs

    """

    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False):

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc

    def simulate(self,
                 generator,
                 sampler,
                 estimator,
                 ):
        """
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator: 
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        def process_one_run(ind_run):
            t_map, m_meta_map, _ = generator.generate()
            m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
            t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
            t_estimated_map = estimator.estimate_map(t_sampled_map_allfreq, mask, m_meta_map)
            v_meta = m_meta_map_all_freqs.flatten()
            v_map = t_map.flatten()
            v_est_map = t_estimated_map.flatten()
            sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
                np.where(v_meta == 0)[0])
            return sq_err_one_runs

        if self.use_parallel_proc:
            num_cores = int(multiprocessing.cpu_count() / 2)
            sq_err_all_runs = Parallel(n_jobs=num_cores)(delayed(process_one_run)(i)
                                                         for i in range(self.n_runs))
            sq_err_all_runs_arr = np.array(sq_err_all_runs)
        else:
            sq_err_all_runs_arr = np.zeros((1, self.n_runs))
            for ind_run in range(self.n_runs):
                sq_err_all_runs_arr[0, ind_run] = process_one_run(ind_run)
        rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
        return rmse

