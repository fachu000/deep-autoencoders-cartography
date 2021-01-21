import numpy as np
import pickle
import sys
import re
import itertools
import time
from Generators.map_generator import MapGenerator
from Generators.gudmundson_map_generator import GudmundsonMapGenerator
from Generators.insite_map_generator import InsiteMapGenerator
from utils.communications import db_to_natural, natural_to_db, natural_to_dbm, db_to_dbm, dbm_to_db, dbm_to_natural
from Samplers.map_sampler import MapSampler
from Estimators.knn_estimator import KNNEstimator
from Estimators.kernel_rr_estimator import KernelRidgeRegrEstimator
from Estimators.group_lasso_multiker import GroupLassoMKEstimator
from Estimators.gaussian_proc_regr import GaussianProcessRegrEstimator
from Estimators.matrix_compl_estimator import MatrixComplEstimator
from Estimators.bem_centralized_lasso import BemCentralizedLassoKEstimator
from Simulators.simulator import Simulator
from Estimators.autoencoder_estimator import AutoEncoderEstimator, get_layer_activations
import matplotlib
from numpy import linalg as npla
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 15})
import gsim
from gsim.gfigure import GFigure


# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# from tensorflow import keras
# from scipy.interpolate import interp1d
# from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
# import time
# import pandas as pd


class ExperimentSet(gsim.AbstractExperimentSet):

    # Generates Fig. 3 : map reconstruction using the autoencoder estimator with code length 4
    def experiment_1003(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(500)

        # Generator
        v_central_freq = [1.4e9]
        map_generator = GudmundsonMapGenerator(
            tx_power=np.array([[11, 5]]), #dBm
            b_shadowing=False,
            v_central_frequencies=v_central_freq)

        # Sampler
        testing_sampler = MapSampler(std_noise=1)

        #  Autoencoder estimator
        architecture_id = '8toy'
        filters = 27
        code_length = 4
        train_autoencoder = True
        num_maps = 400000
        ve_split_frac = 1
        if not train_autoencoder:
            estimator = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=map_generator.m_basis_functions,
                n_filters=filters,
                weight_file=
                'output/autoencoder_experiments/savedWeights/weights.h5')
        else:

            estimator = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=map_generator.m_basis_functions,
                n_filters=filters)
            # Train
            training_sampler = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)
            history, codes = estimator.train(generator=map_generator,
                                               sampler=training_sampler,
                                               learning_rate=5e-4,
                                               n_super_batches=1,
                                               n_maps=num_maps,
                                               perc_train=0.9,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=1,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # Generate a test map and reconstruct
        map, meta_map, _ = map_generator.generate()
        realization_sampl_fac = [0.05]
        l_recontsructed_maps = []
        l_sampled_maps = []
        l_masks = []

        for ind_sf in range(len(realization_sampl_fac)):
            testing_sampler.v_sampling_factor = realization_sampl_fac[ind_sf]
            sampled_map_in, mask = testing_sampler.sample_map(
                map, meta_map)
            if ind_sf == 0:
                l_sampled_maps += [sampled_map_in[:, :, 0]]
                l_masks += [mask]
            estimated_map = estimator.estimate_map(sampled_map_in, mask, meta_map)
            l_recontsructed_maps += [estimated_map[:, :, 0]]

        ExperimentSet.plot_reconstruction(map_generator.x_length,
                                          map_generator.y_length,
                                          list([map[:, :, 0]]),
                                          l_sampled_maps,
                                          l_masks,
                                          realization_sampl_fac,
                                          meta_map,
                                          l_recontsructed_maps,
                                          exp_num)
        return

    # Generates Figs. 5 and 6: this experiment invokes a simulator (with autoencoder , kriging, multikernel, GPR,
    # and KNN estimators), reconstructs a map from the Wireless Insite dataset, plots the map RMSE as a function of the
    # number of measurements when the training and testing maps are from the Gudmundson data set.
    def experiment_1005(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(510)

        # Generator
        v_central_freq = [1.4e9]
        map_generator = GudmundsonMapGenerator(
            # tx_power=np.tile(np.array([[11, 7], [10, 6]]), (int(np.size(v_central_freq) / 2), 1)), # dBm
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=100,
            v_central_frequencies=v_central_freq)
        # Sampler
        sampling_factor = np.concatenate((np.linspace(0.01, 0.1, 10, endpoint=False),
                           np.linspace(0.1, 0.2, 7)), axis=0)[0:14]
        testing_sampler = MapSampler(std_noise=1)

        # Estimators
        # 1. Autoencoder
        architecture_id = '8'
        filters = 27
        code_length = 64
        train_autoencoder = True
        num_maps = 500000
        ve_split_frac = 1
        if not train_autoencoder:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=map_generator.m_basis_functions,
                n_filters=filters,
                weight_file=
                'output/autoencoder_experiments/savedWeights/weights.h5')
        else:

            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=map_generator.m_basis_functions,
                n_filters=filters)
            # Train
            training_sampler = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)
            history, codes = estimator_1.train(generator=map_generator,
                                               sampler=training_sampler,
                                               learning_rate=5e-4,
                                               n_super_batches=1,
                                               n_maps=num_maps,
                                               perc_train=0.9,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=1,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            ExperimentSet.plot_histograms_of_codes_and_visualize(
                map_generator.x_length,
                map_generator.y_length,
                codes,
                estimator_1.chosen_model,
                exp_num,
            )
            ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # 2. All estimators
        all_estimators = [
            estimator_1,
            KernelRidgeRegrEstimator(map_generator.x_length,
                                     map_generator.y_length),
            GroupLassoMKEstimator(map_generator.x_length,
                                  map_generator.y_length,
                                  str_name="Multikernel-Lapl."),
            GroupLassoMKEstimator(map_generator.x_length,
                                  map_generator.y_length,
                                  str_name="Multikernel-RBF",
                                  use_laplac_kernel=False),
            GaussianProcessRegrEstimator(map_generator.x_length,
                                         map_generator.y_length),
            KNNEstimator(map_generator.x_length,
                         map_generator.y_length)
        ]
        estimators_to_sim = list(range(1, 2))

        # Generate a remcom test map and reconstruct it
        # realization_map_generator = map_generator
        realization_map_generator = InsiteMapGenerator(
            l_file_num=np.arange(1, 43),  # the list is the interval [start, stop)
        )
        realization_sampler = MapSampler()
        map, meta_map, _ = realization_map_generator.generate()
        realization_sampl_fac = [0.05, 0.2]
        l_recontsructed_maps = []
        l_sampled_maps = []
        l_masks = []

        for ind_sf in range(len(realization_sampl_fac)):
            realization_sampler.v_sampling_factor = realization_sampl_fac[ind_sf]
            sampled_map_in, mask = realization_sampler.sample_map(
                map, meta_map)
            if ind_sf == 0:
                l_sampled_maps += [sampled_map_in[:, :, 0]]
                l_masks += [mask]
            estimated_map = estimator_1.estimate_map(sampled_map_in, mask, meta_map)
            l_recontsructed_maps += [estimated_map[:, :, 0]]

        ExperimentSet.plot_reconstruction(realization_map_generator.x_length,
                                          realization_map_generator.y_length,
                                          list([map[:, :, 0]]),
                                          l_sampled_maps,
                                          l_masks,
                                          realization_sampl_fac,
                                          meta_map,
                                          l_recontsructed_maps,
                                          exp_num)
        # exit()
        # Simulation pararameters
        n_runs = 100
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert n_run_estimators <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                        'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), np.size(sampling_factor)))
        labels = []
        l_elapsed_times = []
        for ind_est in range(len(estimators_to_sim)):
            start_time = time.time()
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(len(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=map_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)

            end_time = time.time()
            elapsed_time = end_time - start_time
            l_elapsed_times += [elapsed_time]
            labels += [current_estimator.str_name]

        print('The average run-time of the estimators %s for  %d runs each is' % (estimators_to_sim, n_runs),
              "%s" % (np.array(l_elapsed_times) / n_runs))
        print('The RMSE for all the simulated estimators is %s' % RMSE)
        # quit()

        # Plot results
        G = GFigure(xaxis=np.rint(1024 * sampling_factor),
                    yaxis=RMSE[0, :],
                    xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                    ylabel="RMSE(dB)",
                    legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G.add_curve(xaxis=np.rint(1024 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                            legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        return G

    # Generates Fig. 7: this experiment invokes a simulator (with autoencoder , kriging, multikernel, GPR,
    # and KNN estimators) and plots the map RMSE as a function of the number of measurements with the Wireless Iniste
    # data set .
    def experiment_1007(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(0)

        # Testing generator
        testing_generator = InsiteMapGenerator(
            l_file_num=np.arange(41, 43),  # the list is the interval [start, stop)
                          )

        # Sampler
        sampling_factor = np.concatenate((np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 0.2, 7)),
                                         axis=0)
        testing_sampler = MapSampler()

        # Estimators
        # 1. Autoencoder
        architecture_id = '8'
        filters = 27
        code_length = 64
        train_autoencoder = False
        if not train_autoencoder:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=testing_generator.n_grid_points_x,
                n_pts_y=testing_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=testing_generator.m_basis_functions,
                n_filters=filters,
                weight_file=
                'output/autoencoder_experiments/savedWeights/weights.h5')
        else:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=testing_generator.n_grid_points_x,
                n_pts_y=testing_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=testing_generator.m_basis_functions,
                n_filters=filters)

            # Train
            num_maps = 125000
            ve_split_frac = [0.5, 0.5]
            training_generator = InsiteMapGenerator(
                l_file_num=np.arange(1, 41))
            training_sampler = MapSampler(v_sampling_factor=[0.01, 0.2])

            history, codes = estimator_1.train(generator=training_generator,
                                               sampler=training_sampler,
                                               learning_rate=5e-4,
                                               n_maps=num_maps,
                                               perc_train=0.9,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=10,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            # ExperimentSet.plot_histograms_of_codes_and_visualize(
            #     testing_generator.x_length,
            #     testing_generator.y_length,
            #     codes,
            #     estimator_1.chosen_model,
            #     exp_num,
            # )
            # ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # 2. All estimators
        all_estimators = [
            estimator_1,
            KernelRidgeRegrEstimator(testing_generator.x_length,
                                     testing_generator.y_length),
            GroupLassoMKEstimator(testing_generator.x_length,
                                  testing_generator.y_length,
                                  str_name="Multikernel-Lapl."),
            GroupLassoMKEstimator(testing_generator.x_length,
                                  testing_generator.y_length,
                                  str_name="Multikernel-RBF",
                                  use_laplac_kernel=False),
            GaussianProcessRegrEstimator(testing_generator.x_length,
                                         testing_generator.y_length),
            KNNEstimator(testing_generator.x_length,
                         testing_generator.y_length)
        ]

        # Simulation pararameters
        n_runs = 1000
        estimators_to_sim = list(range(1, 7))
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert len(estimators_to_sim) <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                              'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = []
        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=testing_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            labels += [current_estimator.str_name]

        # Plot results
        G = GFigure(xaxis=np.rint(970 * sampling_factor),  # 970 is the average number of grid points that
                                                           # lie on the street
                    yaxis=RMSE[0, :],
                    xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                    ylabel="RMSE(dB)",
                    legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G.add_curve(xaxis=np.rint(970 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                            legend=labels[ind_plot + 1])
        return G

    # Generates Fig. 8: this experiment invokes a simulator with autoencoder  estimator and plots the map RMSE as
    # a function of the number of measurements with and without transfer learning
    def experiment_1008(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(4000)

        # Generator
        v_central_freq = np.array([1.4e9])
        testing_generator = InsiteMapGenerator(
            l_file_num=np.arange(41, 43),  # the list is the interval [start, stop)
        )

        # Sampler
        sampling_factor = np.concatenate((np.linspace(0.05, 0.1, 6, endpoint=False), np.linspace(0.1, 0.2, 4)),
                                         axis=0)
        testing_sampler = MapSampler()

        # Estimators

        architecture_id = '8'
        filters = 27
        code_length = 64
        num_maps = 1000
        ve_split_frac = [0.5, 0.5]
        n_epochs = 100
        training_generator = InsiteMapGenerator(
            m_basis_functions=testing_generator.m_basis_functions,
            l_file_num=np.arange(1, 41))
        training_sampler = MapSampler(v_sampling_factor=[0.05, 0.2])

        # 1. Autoencoder without transfer learning
        estimator_1 = AutoEncoderEstimator(
            n_pts_x=testing_generator.n_grid_points_x,
            n_pts_y=testing_generator.n_grid_points_y,
            arch_id=architecture_id,
            c_length=code_length,
            bases_vals=testing_generator.m_basis_functions,
            n_filters=filters)

        # Train estimator 1
        history, codes = estimator_1.train(generator=training_generator,
                                           sampler=training_sampler,
                                           learning_rate=5e-4,
                                           # n_super_batches=10,
                                           n_maps=num_maps,
                                           perc_train=0.9,
                                           v_split_frac=ve_split_frac,
                                           n_resamples_per_map=30,
                                           n_epochs=n_epochs)

        # Plot training results: losses and visualize codes if enabled
        # ExperimentSet.plot_histograms_of_codes_and_visualize(
        #     testing_generator.x_length,
        #     testing_generator.y_length,
        #     codes,
        #     estimator_1.chosen_model,
        #     exp_num,
        # )
        # ExperimentSet.plot_train_and_val_losses(history, exp_num)
        estimator_1.str_name = "Without transfer learning"

        # 2. Autoencoder with transfer learning
        estimator_2 = AutoEncoderEstimator(
            n_pts_x=testing_generator.n_grid_points_x,
            n_pts_y=testing_generator.n_grid_points_y,
            arch_id=architecture_id,
            c_length=code_length,
            bases_vals=testing_generator.m_basis_functions,
            n_filters=filters)

        # Pretrain with the Gudmundson data set
        num_maps_0 = 500000
        ve_split_frac_0 = 1
        training_generator_0 = GudmundsonMapGenerator(
            # tx_power=np.tile(np.array([[11, 7], [10, 6]]), (int(np.size(v_central_freq) / 2), 1)), # dBm
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=5000,
            v_central_frequencies=v_central_freq)
        training_sampler_0 = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)
        history, codes = estimator_2.train(generator=training_generator_0,
                                           sampler=training_sampler_0,
                                           learning_rate=5e-4,
                                           n_maps=num_maps_0,
                                           v_split_frac=ve_split_frac_0,
                                           n_resamples_per_map=1,
                                           n_epochs=100)

        # Fine-tune with the Wireless Insite data set
        history, codes = estimator_2.train(generator=training_generator,
                                           sampler=training_sampler,
                                           learning_rate=5e-4,
                                           # n_super_batches=10,
                                           n_maps=num_maps,
                                           perc_train=0.9,
                                           v_split_frac=ve_split_frac,
                                           n_resamples_per_map=30,
                                           n_epochs=n_epochs)
        estimator_2.str_name = "With transfer learning"

        # 2. All estimators
        all_estimators = [estimator_1, estimator_2]


        # Simulation pararameters
        n_runs = 1000
        estimators_to_sim = [1, 2]
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert len(estimators_to_sim) <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                              'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = []
        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=testing_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            labels += [current_estimator.str_name]

        # Plot results
        print(RMSE)
        G = GFigure(
            xaxis=np.rint(testing_generator.n_grid_points_x * testing_generator.n_grid_points_y * sampling_factor),
            yaxis=RMSE[0, :],
            xlabel="Number of measurements, " + r"$\vert \Omega \vert $",
            ylabel="RMSE(dB)",
            legend=labels[0])
        if len(estimators_to_sim) >= 1:
            for ind_plot in range(len(estimators_to_sim) - 1):
                G.add_curve(xaxis=np.rint(
                    testing_generator.n_grid_points_x * testing_generator.n_grid_points_y * sampling_factor),
                    yaxis=RMSE[ind_plot + 1, :], legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        return G

    # Generates Fig. 9: this experiment invokes a simulator (with autoencoder estimator) and plots the map RMSE as
    # a function of the code length.
    def experiment_1009(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        np.random.seed(540)

        # Generator
        v_central_freq = [1.4e9]
        map_generator = GudmundsonMapGenerator(
            tx_power=np.array([[11, 5]]),
            num_precomputed_shadowing_mats=5000,
            v_central_frequencies=v_central_freq)

        # sampler
        sampler = MapSampler(v_sampling_factor=0.1, std_noise=1)

        # Estimator
        architecture_id = '5'
        v_code_length = np.concatenate((np.rint(np.linspace(4, 15, 12)),
                                      np.rint(np.linspace(16, 40, 8))), axis=0)
        filter_groups = 4
        v_filters = [27]*filter_groups + [26]*filter_groups + \
                  [25]*filter_groups + [24]*filter_groups + [23]*filter_groups  # same length as the code length
        assert len(v_code_length) == len(v_filters), 'The length of the "v_code length" vector must be equal to that of ' \
                                                        'the "v_filters" vector'

        # Prepare simulation
        n_runs = 1000
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)
        # run simulation
        RMSE = np.zeros((1, len(v_code_length)))
        for ind_code in range(np.size(v_code_length)):
            estimator = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=v_code_length[ind_code],
                bases_vals=map_generator.m_basis_functions,
                n_filters=v_filters[ind_code])
            # Train
            num_maps = 200000
            ve_split_frac = 1
            history, codes = estimator.train(generator=map_generator,
                                               sampler=sampler,
                                               learning_rate=5e-4,
                                               n_super_batches=1,
                                               n_maps=num_maps,
                                               perc_train=0.9,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=1,
                                               n_epochs=100)
            sampler.v_sampling_factor = 0.1
            RMSE[0, ind_code] = simulator.simulate(
                generator=map_generator,
                sampler=sampler,
                estimator=estimator)
            print('Experiment with  code legnth=%d is done with RMSE %.5f' % (v_code_length[ind_code], RMSE[0, ind_code]))

        # Plot results
        G = GFigure(xaxis=v_code_length,
                    yaxis=RMSE[0, :],
                    xlabel='Code length, ' + r'$N_\lambda$',
                    ylabel="RMSE(dB)")

        return G

    # Generates Fig. 10: this experiment invokes a simulator (with autoencoder estimator) and plots the map RMSE as
    # a function of the number of measurements for autoencoders with dense and convolutional layers
    # close to the bottleneck.
    def experiment_1010(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(500)

        # Generator
        v_central_freq = [1.4e9]
        map_generator = GudmundsonMapGenerator(
            # tx_power=np.tile(np.array([[11, 7], [10, 6]]), (int(np.size(v_central_freq) / 2), 1)), # dBm
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=5000,
            v_central_frequencies=v_central_freq)
        # Sampler
        sampling_factor = np.concatenate((np.linspace(
            0.05, 0.2, 10, endpoint=False), np.linspace(0.2, 0.5, 7)),
            axis=0)
        testing_sampler = MapSampler(std_noise=1)

        # Estimators
        v_architecture_ids = ['5', '8']
        v_filters = [27, 34]
        code_length = 64
        num_maps = 500000
        ve_split_frac = 1
        v_sampl_factor = [0.05, 0.5]

        # 1. Autoencoder with dense layers
        estimator_1 = AutoEncoderEstimator(
            n_pts_x=map_generator.n_grid_points_x,
            n_pts_y=map_generator.n_grid_points_y,
            arch_id=v_architecture_ids[0],
            c_length=code_length,
            bases_vals=map_generator.m_basis_functions,
            n_filters=v_filters[0])
        # Train
        training_sampler = MapSampler(v_sampling_factor=v_sampl_factor, std_noise=1)
        history, codes = estimator_1.train(generator=map_generator,
                                           sampler=training_sampler,
                                           learning_rate=5e-4,
                                           n_super_batches=1,
                                           n_maps=num_maps,
                                           perc_train=0.9,
                                           v_split_frac=ve_split_frac,
                                           n_resamples_per_map=1,
                                           n_epochs=100)
        estimator_1.str_name = "Fully connected"

        # 2. Fully convolutional  autoencoders
        estimator_2 = AutoEncoderEstimator(
            n_pts_x=map_generator.n_grid_points_x,
            n_pts_y=map_generator.n_grid_points_y,
            arch_id=v_architecture_ids[1],
            c_length=code_length,
            bases_vals=map_generator.m_basis_functions,
            n_filters=v_filters[1])
        # Train
        training_sampler = MapSampler(v_sampling_factor=v_sampl_factor, std_noise=1)
        history, codes = estimator_2.train(generator=map_generator,
                                           sampler=training_sampler,
                                           learning_rate=5e-4,
                                           n_super_batches=1,
                                           n_maps=num_maps,
                                           perc_train=0.9,
                                           v_split_frac=ve_split_frac,
                                           n_resamples_per_map=1,
                                           n_epochs=10)
        estimator_2.str_name = "Convolutional"

        # Plot training results: losses and visualize codes if enabled
        # ExperimentSet.plot_histograms_of_codes_and_visualize(
        #     map_generator.x_length,
        #     map_generator.y_length,
        #     codes,
        #     estimator_1.chosen_model,
        #     exp_num,
        # )
        # ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # 2. All estimators
        all_estimators = [
            estimator_1,
            estimator_2
        ]
        estimators_to_sim = [1, 2]

        # Simulation pararameters
        n_runs = 1000
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert n_run_estimators <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                        'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = []
        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=map_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            labels += [current_estimator.str_name]
            # Plot results
        print(RMSE)

        G = GFigure(xaxis=np.rint(1024 * sampling_factor),
                    yaxis=RMSE[0, :],
                    xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                    ylabel="RMSE(dB)",
                    legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G.add_curve(xaxis=np.rint(1024 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                            legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        return G

    # Generates Fig. 11: this experiment invokes a simulator (with autoencoder estimator) and plots the map RMSE as
    # a function of the number of layers for two types of activation functions.
    def experiment_1011(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        np.random.seed(540)

        # Generator
        v_central_freq = [1.4e9]
        map_generator = GudmundsonMapGenerator(
            # tx_power=np.tile(np.array([[11, 7], [10, 6]]), (int(np.size(v_central_freq) / 2), 1)), # dBm
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=20,
            v_central_frequencies=v_central_freq)

        # sampler
        testing_sampler = MapSampler(v_sampling_factor=0.1, std_noise=1)

        # Estimator
        v_architecture_ids = ['1','2', '3', '4', '5', '6', '7']
        code_length = 64
        v_filters = [18, 17, 35, 30, 27, 30, 27]   # adjusted so the network has the same number of parameters
        assert len(v_architecture_ids) == len(v_filters), 'The length of the list "v_architecture_ids" must be equal to ' \
                                                      'that of the "v_filters" vector'
        activ_functions = ['prelu', 'leakyrelu']

        # Prepare simulation
        n_runs = 1000
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)
        # run simulation
        n_layers = np.zeros((1, len(v_architecture_ids)))
        RMSE = np.zeros((len(activ_functions), len(v_architecture_ids)))
        for ind_activ_func in range(len(activ_functions)):
            for ind_arch in range(np.size(v_architecture_ids)):
                estimator = AutoEncoderEstimator(
                    n_pts_x=map_generator.n_grid_points_x,
                    n_pts_y=map_generator.n_grid_points_y,
                    arch_id=v_architecture_ids[ind_arch],
                    c_length=code_length,
                    bases_vals=map_generator.m_basis_functions,
                    n_filters=v_filters[ind_arch],
                    activ_func_name=activ_functions[ind_activ_func])
                # Train
                num_maps = 400000
                ve_split_frac = 1
                training_sampler = MapSampler(v_sampling_factor=[0.05, 0.3], std_noise=1)
                history, codes = estimator.train(generator=map_generator,
                                                 sampler=training_sampler,
                                                 learning_rate=5e-4,
                                                 n_super_batches=1,
                                                 n_maps=num_maps,
                                                 perc_train=0.9,
                                                 v_split_frac=ve_split_frac,
                                                 n_resamples_per_map=1,
                                                 n_epochs=100)
                if ind_activ_func == 0:
                    encoder = estimator.chosen_model.get_layer('encoder')
                    decoder = estimator.chosen_model.get_layer('decoder')
                    n_layers[0, ind_arch] = len(encoder.layers) + len(
                        decoder.layers) - 12  # 12 = 1 for reshape + 2 input + 1 flatten layers + 8 tensorflow ops
                testing_sampler.v_sampling_factor = 0.3
                RMSE[ind_activ_func, ind_arch] = simulator.simulate(
                    generator=map_generator,
                    sampler=testing_sampler,
                    estimator=estimator)
                print('Experiment with activation function type_%d and architecture_%s is done with RMSE %.5f'
                      % (ind_activ_func + 1, v_architecture_ids[ind_arch], RMSE[ind_activ_func, ind_arch]))

        # Plot results
        labels = ['With PReLU', 'With LeakyReLU']
        G = GFigure(xaxis=n_layers,
                    yaxis=RMSE[0, :],
                    xlabel='Number of layers'+' ' + r'$L$',
                    ylabel='RMSE(dB)',
                    legend=labels[0])
        if len(activ_functions) > 1:
            for ind_plot in range(len(activ_functions) - 1):
                G.add_curve(xaxis=n_layers, yaxis=RMSE[ind_plot + 1, :],
                            legend=labels[ind_plot + 1])
        return G

    # Generates Fig. 12: this experiment  visualizes the decoder outputs when latent variables are provided
    # at the input of the trained decoder for 2 scenarios: path loss (Fig. 12a) and shadowing (Fig. 12b)
    def experiment_1012(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(500)

        # Generators
        v_central_freq = [1.4e9]

        #  generator for path loss only scenario
        map_generator_pathloss = GudmundsonMapGenerator(
            tx_power_interval=[5, 11],  # dBm
            num_precomputed_shadowing_mats=5000,
            v_central_frequencies=v_central_freq)

        #  generator for scenario with shadowing
        map_generator_shadowing = GudmundsonMapGenerator(
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=5000,
            v_central_frequencies=v_central_freq)

        # Estimators
        filters = 27
        num_maps = 500000
        n_epochs = 100
        ve_split_frac = 1
        training_sampler = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)

        # train with path loss only
        architecture_id_pl = '5'
        code_length_pl = 4

        estimator_1 = AutoEncoderEstimator(
            n_pts_x=map_generator_pathloss.n_grid_points_x,
            n_pts_y=map_generator_pathloss.n_grid_points_y,
            arch_id=architecture_id_pl,
            c_length=code_length_pl,
            bases_vals=map_generator_pathloss.m_basis_functions,
            n_filters=filters)
        # Train
        history, codes_pl = estimator_1.train(generator=map_generator_pathloss,
                                           sampler=training_sampler,
                                           learning_rate=5e-4,
                                           n_super_batches=1,
                                           n_maps=num_maps,
                                           perc_train=0.9,
                                           v_split_frac=ve_split_frac,
                                           n_resamples_per_map=1,
                                           n_epochs=n_epochs)

        # train with shadowing
        architecture_id_sh = '5'
        code_length_sh = 64

        # Autoencoder for path loss only
        estimator_2 = AutoEncoderEstimator(
            n_pts_x=map_generator_shadowing.n_grid_points_x,
            n_pts_y=map_generator_shadowing.n_grid_points_y,
            arch_id=architecture_id_sh,
            c_length=code_length_sh,
            bases_vals=map_generator_shadowing.m_basis_functions,
            n_filters=filters)
        # Train
        history, codes_sh = estimator_2.train(generator=map_generator_shadowing,
                                              sampler=training_sampler,
                                               learning_rate=5e-4,
                                               n_super_batches=1,
                                               n_maps=num_maps,
                                               perc_train=0.9,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=1,
                                               n_epochs=n_epochs)

        #  Generates Fig. 14a
        ExperimentSet.plot_histograms_of_codes_and_visualize(
            map_generator_pathloss.x_length,
            map_generator_pathloss.y_length,
            codes_pl,
            estimator_1.chosen_model,
            exp_num,
            visualize=True,
            use_cov_matr=False)

        #  Generates Fig. 14b
        ExperimentSet.plot_histograms_of_codes_and_visualize(
            map_generator_shadowing.x_length,
            map_generator_shadowing.y_length,
            codes_sh,
            estimator_2.chosen_model,
            exp_num,
            visualize=True)
        
        # ExperimentSet.plot_train_and_val_losses(history, exp_num)
        return

    # Generates Figs. 13, 14, and 16 for Gaussian functions: this experiment invokes a simulator (with DL and
    # Centralized lasso  estimators) and plots the map RMSE as a function of the number of measurements (BEM along
    # the frequency domain). The network is trained and tested over the Gudmundson data set
    def experiment_1013(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        np.random.seed(4007)

        # Generator
        v_central_freq = np.linspace(1.4e9, 1.45e9, 6)[1:4]
        v_sampled_freq = np.linspace(1.4e9, 1.45e9, 42)[5:37]
        bandwidth = 2e7  # MHz

        m_basis_functions = MapGenerator.generate_bases(
            v_central_frequencies=v_central_freq,
            v_sampled_frequencies=v_sampled_freq,
            fun_base_function=lambda freq: MapGenerator.gaussian_base(freq, bandwidth / 4),
            b_noise_function=True,
        )

        map_generator = GudmundsonMapGenerator(
            m_basis_functions=m_basis_functions,
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=100,
            v_central_frequencies=v_central_freq,
            noise_power_interval=[-100, -90])  # dBm

        # Sampler
        # sampling_factor = [0.05]
        sampling_factor = [0.05] # np.concatenate((np.linspace(0.05, 0.2, 10, endpoint=False), np.linspace(0.2, 0.45, 7)), axis=0)
        testing_sampler = MapSampler(std_noise=1)

        # Estimators
        # 1. Autoencoder
        architecture_id = '8'
        filters = 32
        code_length = 64
        train_autoencoder = False
        num_maps = 25000
        ve_split_frac = 1
        if not train_autoencoder:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=m_basis_functions,
                n_filters=filters,
                weight_file=
                'output/autoencoder_experiments/savedWeights/weights.h5')
        else:

            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=m_basis_functions,
                n_filters=filters)
            # Train
            training_sampler = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)
            history, codes = estimator_1.train(generator=map_generator,
                                               sampler=training_sampler,
                                               learning_rate=1e-4,
                                               n_super_batches=10,
                                               n_maps=num_maps,
                                               perc_train=0.96,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=5,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            # ExperimentSet.plot_histograms_of_codes_and_visualize(
            #     map_generator.x_length,
            #     map_generator.y_length,
            #     codes,
            #     estimator_1.chosen_model,
            #     exp_num,
            # )
            # ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # Plot training coefficients
        # b_show_training_coeff = False
        # if b_show_training_coeff:
        #     plot_training_coeficients(map_generator, estimator_1, exp_num)

        # 2. All estimators
        all_estimators = [BemCentralizedLassoKEstimator(x_length=map_generator.x_length,
                                                        y_length=map_generator.y_length,
                                                        n_grid_points_x=map_generator.n_grid_points_x,
                                                        n_grid_points_y=map_generator.n_grid_points_y,
                                                        bases_vals=m_basis_functions[0:len(v_central_freq), :]),
                          estimator_1]
        estimators_to_sim = [1, 2]
        G = []

        # Show realization
        b_show_realization = False
        if b_show_realization:

            t_true_map, m_meta_map, t_channel_pow = map_generator.generate()
            testing_sampler.v_sampling_factor = 0.5
            sampled_map, mask = testing_sampler.sample_map(t_true_map, m_meta_map)

            # obtain PSD estimates at random point
            rand_pt = np.random.choice(map_generator.n_grid_points_x, size=2)
            l_all_psds = [t_true_map[rand_pt[0], rand_pt[1], :]]

            # True BEM coefficient maps
            l_bem_maps = [t_channel_pow]
            reconst_labels = ['True']

            # BEM products
            b_show_bem_products = True
            l_bem_prods = []

            for ind_estimator in range(len(all_estimators)):

                # PSD at the random point
                estimated_map = all_estimators[ind_estimator].estimate_map(sampled_map, mask, m_meta_map)
                l_all_psds += [estimated_map[rand_pt[0], rand_pt[1], :]]
                reconst_labels += [all_estimators[ind_estimator].str_name]

                # BEM coefficient map
                current_bem_map = all_estimators[ind_estimator].estimate_bem_coefficient_map(sampled_map, mask,
                                                                                             m_meta_map)
                l_bem_maps += [current_bem_map]

                # BEM products at the random point
                if b_show_bem_products:
                    bem_map_nat = db_to_natural(current_bem_map)
                    current_bem_prod = []
                    for ind_base in range(current_bem_map.shape[2]):
                        current_bem_prod += [natural_to_db(bem_map_nat[rand_pt[0], rand_pt[1], ind_base] *
                                                           m_basis_functions[ind_base, :])]
                    l_bem_prods += [np.array(current_bem_prod)]

            # interpolate and plot psd estimates
            all_psds = db_to_dbm(np.array(l_all_psds))  # psd in dBm
            print('The estimated PSDs are:\n')
            print(all_psds)
            v_sampled_freq_mhz = v_sampled_freq / 1e6
            G1 = GFigure(xaxis=v_sampled_freq_mhz,
                         yaxis=all_psds[0, :],
                         xlabel='f [MHz]',
                         ylabel=r"$ \Psi(\mathbf{x},f) [dBm] $",
                         legend=reconst_labels[0])
            for ind_plot in range(all_psds.shape[0] - 1):
                G1.add_curve(xaxis=v_sampled_freq_mhz, yaxis=all_psds[ind_plot + 1, :],
                             legend=reconst_labels[ind_plot + 1])

            # interpolate and plot bem products
            if b_show_bem_products:
                bem_prods = db_to_dbm(np.array(l_bem_prods))  # bem products in dBm
                for ind_est in range(bem_prods.shape[0]):
                    for ind_plot in range(bem_prods.shape[1]):
                        G1.add_curve(xaxis=v_sampled_freq_mhz,
                                     yaxis=bem_prods[ind_est, ind_plot, :])
            G.append(G1)

            # plot BEM coefficient estimates
            bases_to_plot = np.arange(l_bem_maps[0].shape[2])
            ExperimentSet.plot_b_coefficients(map_generator.x_length,
                                              map_generator.y_length,
                                              np.array(l_bem_maps),
                                              bases_to_plot,
                                              reconst_labels,
                                              exp_num,
                                              file_name='True_and_Estimated_bcoeffs')

        # Simulation pararameters
        n_runs = 100
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert n_run_estimators <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                        'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = []
        l_elapsed_times = []
        for ind_est in range(len(estimators_to_sim)):
            start_time = time.time()
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=map_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            end_time = time.time()
            elapsed_time = end_time - start_time
            l_elapsed_times += [elapsed_time]
            labels += [current_estimator.str_name]

        print('The average run-time of the estimators %s for  %d runs each is' % (estimators_to_sim, n_runs),
              "%s" % (np.array(l_elapsed_times) / n_runs))
        print('The RMSE for all the simulated estimators is %s' % RMSE)
        quit()

        # Plot results
        G2 = GFigure(xaxis=np.rint(1024 * sampling_factor),
                     yaxis=RMSE[0, :],
                     xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                     ylabel="RMSE(dB)",
                     legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G2.add_curve(xaxis=np.rint(1024 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                             legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        G.append(G2)
        return G

    # Generates Figs. 15 and 16 for raised-cosine functions: this experiment invokes a simulator(with DL and
    # Centralized lasso  estimators) and plots the map RMSE as a function of the number of measurements (BEM along
    # the frequency domain). The network is trained and tested over the Gudmundson data set
    def experiment_1015(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(4007)

        # Generator
        v_central_freq = np.linspace(1.4e9, 1.45e9, 6)[1:4]
        v_sampled_freq = np.linspace(1.4e9, 1.45e9, 42)[5:37]
        bandwidth = 2e7  # MHz
        roll_off_fac = 0.4  # for raised cosine base

        m_basis_functions = MapGenerator.generate_bases(
            v_central_frequencies=v_central_freq,
            v_sampled_frequencies=v_sampled_freq,
            fun_base_function=lambda freq: MapGenerator.raised_cosine_base(freq, roll_off_fac, bandwidth / 2),
            b_noise_function=True,
        )

        map_generator = GudmundsonMapGenerator(
            m_basis_functions=m_basis_functions,
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=3000,
            v_central_frequencies=v_central_freq,
            noise_power_interval=[-100, -90])  # dBm

        # Sampler
        # sampling_factor = [0.05]
        sampling_factor = np.concatenate((np.linspace(0.05, 0.1, 1, endpoint=False), np.linspace(0.1, 0.2, 1)),
                                         axis=0)
        testing_sampler = MapSampler(std_noise=1)

        # Estimators
        # 1. Autoencoder
        architecture_id = '8'
        filters = 32
        code_length = 64
        train_autoencoder = True
        num_maps = 25000
        ve_split_frac = 1
        if not train_autoencoder:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=m_basis_functions,
                n_filters=filters,
                weight_file=
                'output/autoencoder_experiments/savedWeights/weights.h5')
        else:

            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=m_basis_functions,
                n_filters=filters)
            # Train
            training_sampler = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)
            history, codes = estimator_1.train(generator=map_generator,
                                               sampler=training_sampler,
                                               learning_rate=1e-4,
                                               n_super_batches=10,
                                               n_maps=num_maps,
                                               perc_train=0.96,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=5,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            # ExperimentSet.plot_histograms_of_codes_and_visualize(
            #     map_generator.x_length,
            #     map_generator.y_length,
            #     codes,
            #     estimator_1.chosen_model,
            #     exp_num,
            # )
            # ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # Plot training coefficients
        # b_show_training_coeff = False
        # if b_show_training_coeff:
        #     plot_training_coeficients(map_generator, estimator_1, exp_num)

        # 2. All estimators
        all_estimators = [BemCentralizedLassoKEstimator(x_length=map_generator.x_length,
                                                        y_length=map_generator.y_length,
                                                        n_grid_points_x=map_generator.n_grid_points_x,
                                                        n_grid_points_y=map_generator.n_grid_points_y,
                                                        bases_vals=m_basis_functions[0:len(v_central_freq), :]),
                          estimator_1]
        estimators_to_sim = [1]
        G = []

        # Show realization
        b_show_realization = True
        if b_show_realization:

            t_true_map, m_meta_map, t_channel_pow = map_generator.generate()
            testing_sampler.v_sampling_factor = 0.5
            sampled_map, mask = testing_sampler.sample_map(t_true_map, m_meta_map)

            # obtain PSD estimates at random point
            rand_pt = np.random.choice(map_generator.n_grid_points_x, size=2)
            l_all_psds = [t_true_map[rand_pt[0], rand_pt[1], :]]

            # True BEM coefficient maps
            l_bem_maps = [t_channel_pow]
            reconst_labels = ['True']

            # BEM products
            b_show_bem_products = False
            l_bem_prods = []

            for ind_estimator in range(len(all_estimators)):

                # PSD at the random point
                estimated_map = all_estimators[ind_estimator].estimate_map(sampled_map, mask, m_meta_map)
                l_all_psds += [estimated_map[rand_pt[0], rand_pt[1], :]]
                reconst_labels += [all_estimators[ind_estimator].str_name]

                # BEM coefficient map
                current_bem_map = all_estimators[ind_estimator].estimate_bem_coefficient_map(sampled_map, mask,
                                                                                             m_meta_map)
                l_bem_maps += [current_bem_map]

                # BEM products at the random point
                if b_show_bem_products:
                    bem_map_nat = db_to_natural(current_bem_map)
                    current_bem_prod = []
                    for ind_base in range(current_bem_map.shape[2]):
                        current_bem_prod += [natural_to_db(bem_map_nat[rand_pt[0], rand_pt[1], ind_base] *
                                                           m_basis_functions[ind_base, :])]
                    l_bem_prods += [np.array(current_bem_prod)]

            # interpolate and plot psd estimates
            all_psds = db_to_dbm(np.array(l_all_psds))  # psd in dBm
            print('The estimated PSDs are:\n')
            print(all_psds)
            v_sampled_freq_mhz = v_sampled_freq / 1e6
            G1 = GFigure(xaxis=v_sampled_freq_mhz,
                         yaxis=all_psds[0, :],
                         xlabel='f [MHz]',
                         ylabel=r"$ \Psi(\mathbf{x},f) [dBm] $",
                         legend=reconst_labels[0])
            for ind_plot in range(all_psds.shape[0] - 1):
                G1.add_curve(xaxis=v_sampled_freq_mhz, yaxis=all_psds[ind_plot + 1, :],
                             legend=reconst_labels[ind_plot + 1])

            # interpolate and plot bem products
            if b_show_bem_products:
                bem_prods = db_to_dbm(np.array(l_bem_prods))  # bem products in dBm
                for ind_est in range(bem_prods.shape[0]):
                    for ind_plot in range(bem_prods.shape[1]):
                        G1.add_curve(xaxis=v_sampled_freq_mhz,
                                     yaxis=bem_prods[ind_est, ind_plot, :])
            G.append(G1)

            # plot BEM coefficient estimates
            bases_to_plot = np.arange(l_bem_maps[0].shape[2])
            ExperimentSet.plot_b_coefficients(map_generator.x_length,
                                              map_generator.y_length,
                                              np.array(l_bem_maps),
                                              bases_to_plot,
                                              reconst_labels,
                                              exp_num,
                                              file_name='True_and_Estimated_bcoeffs')

        # Simulation pararameters
        n_runs = 1000
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=True)

        # run the simulation
        assert n_run_estimators <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                        'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = []
        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=map_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            labels += [current_estimator.str_name]
            # Plot results
        print(RMSE)

        G2 = GFigure(xaxis=np.rint(1024 * sampling_factor),
                     yaxis=RMSE[0, :],
                     xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                     ylabel="RMSE(dB)",
                     legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G2.add_curve(xaxis=np.rint(1024 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                             legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        G.append(G2)
        return G

    # Generates Fig. 17: this experiment invokes a simulator(with DL estimator) and plots the map RMSE as a function of
    # the number of measurements (BEM along the frequency domain). The network is trained and tested over the Wireless
    # Insite data set
    def experiment_1017(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(4000)

        # Generator
        v_sampled_freq = np.linspace(1.4e9, 1.45e9, 42)[5:37]
        roll_off_fac = 0.4  # for raised cosine base
        noise_p_interval = [-180, -170]
        v_bases = [4, 7, 13]
        v_bandwidth =[2e7, 1.3e7, 6.5e6]  # MHz
        assert len(v_bases) == len(v_bandwidth), 'The length of the "v_bases" vector must be equal to ' \
                                                      'that of the "v_bandwidth" vector'

        # Sampler
        sampling_factor = np.concatenate((np.linspace(0.05, 0.2, 10, endpoint=False), np.linspace(0.2, 0.45, 7)),
                                         axis=0)
        testing_sampler = MapSampler()

        RMSE = np.zeros((len(v_bases), len(sampling_factor)))
        labels = []
        for ind_base in range(len(v_bases)):
            v_central_freq = np.linspace(1.4e9, 1.45e9, v_bases[ind_base]+1)[1:v_bases[ind_base]]
            m_basis_functions = MapGenerator.generate_bases(
                v_central_frequencies=v_central_freq,
                v_sampled_frequencies=v_sampled_freq,
                fun_base_function=lambda freq: MapGenerator.raised_cosine_base(freq, roll_off_fac,
                                                                               v_bandwidth[ind_base] / 2),
                b_noise_function=True)
            testing_generator = InsiteMapGenerator(
                m_basis_functions=m_basis_functions,
                l_file_num=np.arange(41, 43),  # the list is the interval [start, stop)
                noise_power_interval=noise_p_interval)

            # 1. Autoencoder estimator
            architecture_id = '8'
            filters = 32
            code_length = 64
            train_autoencoder = False
            estimator = AutoEncoderEstimator(
                n_pts_x=testing_generator.n_grid_points_x,
                n_pts_y=testing_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=m_basis_functions,
                n_filters=filters)

            # Train
            num_maps = 25000
            ve_split_frac = [0.5, 0.5]
            training_sampler = MapSampler(v_sampling_factor=[0.05, 0.45])
            training_generator = InsiteMapGenerator(
                m_basis_functions=m_basis_functions,
                l_file_num=np.arange(1, 41),
                noise_power_interval=noise_p_interval)

            history, codes = estimator.train(generator=training_generator,
                                               sampler=training_sampler,
                                               learning_rate=5e-4,
                                               n_super_batches=10,
                                               n_maps=num_maps,
                                               perc_train=0.96,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=5,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            # ExperimentSet.plot_histograms_of_codes_and_visualize(
            #     testing_generator.x_length,
            #     testing_generator.y_length,
            #     codes,
            #     estimator_1.chosen_model,
            #     exp_num,
            # )
            # ExperimentSet.plot_train_and_val_losses(history, exp_num)



            # Simulation pararameters
            n_runs = 1000

            simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

            # run the simulation  for each base value in v_bases
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_base, ind_sampling] = simulator.simulate(
                    generator=testing_generator,
                    sampler=testing_sampler,
                    estimator=estimator)
            labels += ['B = %d' % v_bases[ind_base]]

        # Plot results
        print(RMSE)
        G = GFigure(
            xaxis=np.rint(970 * sampling_factor),
            yaxis=RMSE[0, :],
            xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
            ylabel="RMSE(dB)",
            legend=labels[0])
        if len(v_bases) >= 1:
            for ind_plot in range(len(v_bases) - 1):
                G.add_curve(xaxis=np.rint(
                    970 * sampling_factor),
                    yaxis=RMSE[ind_plot + 1, :], legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        return G

    # this experiment invokes a simulator (with autoencoder and matrix completion estimators) and
    # plots the map RMSE as a function of the number of measurements with the Gudmundson data set.
    def experiment_1018(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        np.random.seed(500)

        # Generator
        v_central_freq = [1.4e9]
        map_generator = GudmundsonMapGenerator(
            # tx_power=np.tile(np.array([[11, 7], [10, 6]]), (int(np.size(v_central_freq) / 2), 1)), # dBm
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=5000,
            v_central_frequencies=v_central_freq)
        # Sampler
        sampling_factor = np.linspace(0.3, 0.8, 12)
        testing_sampler = MapSampler(std_noise=1)

        # Estimators
        # 1. Autoencoder
        architecture_id = '8'
        filters = 27
        code_length = 64
        train_autoencoder = True
        num_maps = 500000
        ve_split_frac = 1
        if not train_autoencoder:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=map_generator.m_basis_functions,
                n_filters=filters,
                weight_file=
                'output/autoencoder_experiments/savedWeights/weights.h5')
        else:

            estimator_1 = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=map_generator.m_basis_functions,
                n_filters=filters)
            # Train
            training_sampler = MapSampler(v_sampling_factor=[0.3, 0.8], std_noise=1)
            history, codes = estimator_1.train(generator=map_generator,
                                               sampler=training_sampler,
                                               learning_rate=1e-4,
                                               n_super_batches=1,
                                               n_maps=num_maps,
                                               perc_train=0.9,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=1,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            ExperimentSet.plot_histograms_of_codes_and_visualize(
                map_generator.x_length,
                map_generator.y_length,
                codes,
                estimator_1.chosen_model,
                exp_num,
            )
            ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # 2. All estimators
        all_estimators = [
            estimator_1,
            MatrixComplEstimator(tau=1e-4)
        ]
        estimators_to_sim = [1, 2]

        # Simulation pararameters
        n_runs = 1000
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert n_run_estimators <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                        'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = []
        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=map_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            labels += [current_estimator.str_name]
            # Plot results
        print(RMSE)

        G = GFigure(xaxis=np.rint(1024 * sampling_factor),
                    yaxis=RMSE[0, :],
                    xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                    ylabel="RMSE(dB)",
                    legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G.add_curve(xaxis=np.rint(1024 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                            legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        return G

    # this experiment invokes a simulator (with autoencoder and matrix completion estimators) and
    # plots the map RMSE as a function of the number of measurements with the Wireless Iniste data set .
    def experiment_1019(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(0)

        # Testing generator
        testing_generator = InsiteMapGenerator(
            l_file_num=np.arange(41, 43),  # the list is the interval [start, stop)
        )

        # Sampler
        sampling_factor = np.linspace(0.3, 0.8, 12)
        testing_sampler = MapSampler()

        # Estimators
        # 1. Autoencoder
        architecture_id = '8'
        filters = 27
        code_length = 64
        train_autoencoder = True
        if not train_autoencoder:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=testing_generator.n_grid_points_x,
                n_pts_y=testing_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=testing_generator.m_basis_functions,
                n_filters=filters,
                weight_file=
                'output/autoencoder_experiments/savedWeights/weights.h5')
        else:
            estimator_1 = AutoEncoderEstimator(
                n_pts_x=testing_generator.n_grid_points_x,
                n_pts_y=testing_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=testing_generator.m_basis_functions,
                n_filters=filters)

            # Train
            num_maps = 125000
            ve_split_frac = [0.5, 0.5]
            training_generator = InsiteMapGenerator(
                l_file_num=np.arange(1, 41))
            training_sampler = MapSampler(v_sampling_factor=[0.05, 0.45])

            history, codes = estimator_1.train(generator=training_generator,
                                               sampler=training_sampler,
                                               learning_rate=5e-4,
                                               n_maps=num_maps,
                                               perc_train=0.9,
                                               v_split_frac=ve_split_frac,
                                               n_resamples_per_map=10,
                                               n_epochs=100)

            # Plot training results: losses and visualize codes if enabled
            # ExperimentSet.plot_histograms_of_codes_and_visualize(
            #     testing_generator.x_length,
            #     testing_generator.y_length,
            #     codes,
            #     estimator_1.chosen_model,
            #     exp_num,
            # )
            # ExperimentSet.plot_train_and_val_losses(history, exp_num)

        # 2. All estimators
        all_estimators = [
            estimator_1,
            MatrixComplEstimator(tau=1e-4)
        ]

        # Simulation pararameters
        n_runs = 1000
        estimators_to_sim = [1, 2]
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert len(estimators_to_sim) <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                              'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = []
        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=testing_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            labels += [current_estimator.str_name]

        # Plot results
        G = GFigure(xaxis=np.rint(970 * sampling_factor),  # 970 is the average number of grid points that
                    # lie on the street
                    yaxis=RMSE[0, :],
                    xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                    ylabel="RMSE(dB)",
                    legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G.add_curve(xaxis=np.rint(1024 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                            legend=labels[ind_plot + 1])
        return G

    # Generates Fig...... for Gaussian functions: this experiment invokes a simulator with DL
    # estimator and plots the map RMSE as a function of the number of measurements (weight sharing along
    # the frequency domain). The network is trained and tested over the Gudmundson data set
    def experiment_1020(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        np.random.seed(4007)

        # Generator
        v_central_freq = np.linspace(1.4e9, 1.45e9, 6)[1:4]
        v_sampled_freq = np.linspace(1.4e9, 1.45e9, 42)[5:37]
        bandwidth = 2e7  # MHz

        m_basis_functions = MapGenerator.generate_bases(
            v_central_frequencies=v_central_freq,
            v_sampled_frequencies=v_sampled_freq,
            fun_base_function=lambda freq: MapGenerator.gaussian_base(freq, bandwidth / 4),
            b_noise_function=True,
        )

        map_generator = GudmundsonMapGenerator(
            m_basis_functions=m_basis_functions,
            tx_power_interval=[5, 11],  # dBm
            b_shadowing=True,
            num_precomputed_shadowing_mats=3000,
            v_central_frequencies=v_central_freq,
            noise_power_interval=[-100, -90])  # dBm

        # Sampler
        # sampling_factor = [0.05]
        sampling_factor = np.concatenate((np.linspace(0.05, 0.1, 10, endpoint=False), np.linspace(0.1, 0.2, 7)),
                                         axis=0)
        testing_sampler = MapSampler(std_noise=1)

        # Estimators
        architecture_id = '8'
        filters = 32
        code_length = 64
        num_maps = 125000
        ve_split_frac = 1
        n_estimators = 2  # one for estimating separately for each freq(weight sharing) sharing and another with BEM
        all_estimators = []
        for ind_est in range(n_estimators):
            b_estimate_separt = False
            if ind_est == 0:
                b_estimate_separt = True
            estimator = AutoEncoderEstimator(
                n_pts_x=map_generator.n_grid_points_x,
                n_pts_y=map_generator.n_grid_points_y,
                arch_id=architecture_id,
                c_length=code_length,
                bases_vals=m_basis_functions,
                n_filters=filters,
                est_separately_accr_freq=b_estimate_separt)
            # Train
            training_sampler = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)
            history, codes = estimator.train(generator=map_generator,
                                             sampler=training_sampler,
                                             learning_rate=1e-4,
                                             n_super_batches=6,
                                             n_maps=num_maps,
                                             perc_train=0.96,
                                             v_split_frac=ve_split_frac,
                                             n_resamples_per_map=1,
                                             n_epochs=50)
            # Plot training results: losses and visualize codes if enabled
            # ExperimentSet.plot_histograms_of_codes_and_visualize(
            #     map_generator.x_length,
            #     map_generator.y_length,
            #     codes,
            #     estimator_1.chosen_model,
            #     exp_num,
            # )
            # ExperimentSet.plot_train_and_val_losses(history, exp_num)
            all_estimators += [estimator]

        estimators_to_sim = [1, 2]
        G = []

        # Show realization
        b_show_realization = False
        if b_show_realization:

            t_true_map, m_meta_map, t_channel_pow = map_generator.generate()
            testing_sampler.v_sampling_factor = 0.5
            sampled_map, mask = testing_sampler.sample_map(t_true_map, m_meta_map)

            # obtain PSD estimates at random point
            rand_pt = np.random.choice(map_generator.n_grid_points_x, size=2)
            l_all_psds = [t_true_map[rand_pt[0], rand_pt[1], :]]

            # True BEM coefficient maps
            reconst_labels = ['True']
            for ind_estimator in range(len(estimators_to_sim)):
                current_estimator = all_estimators[estimators_to_sim[ind_estimator] - 1]
                estimated_map = current_estimator.estimate_map(sampled_map, mask, m_meta_map)
                l_all_psds += [estimated_map[rand_pt[0], rand_pt[1], :]]
                reconst_labels += [current_estimator.str_name]


            # interpolate and plot psd estimates
            all_psds = db_to_dbm(np.array(l_all_psds))  # psd in dBm
            print('The estimated PSDs are:\n')
            print(all_psds)
            v_sampled_freq_mhz = v_sampled_freq / 1e6
            G1 = GFigure(xaxis=v_sampled_freq_mhz,
                         yaxis=all_psds[0, :],
                         xlabel='f [MHz]',
                         ylabel=r"$ \Psi(\mathbf{x},f) [dBm] $",
                         legend=reconst_labels[0])
            for ind_plot in range(all_psds.shape[0] - 1):
                G1.add_curve(xaxis=v_sampled_freq_mhz, yaxis=all_psds[ind_plot + 1, :],
                             legend=reconst_labels[ind_plot + 1])

            G.append(G1)

        # Simulation pararameters
        n_runs = 1000
        n_run_estimators = len(estimators_to_sim)
        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        assert n_run_estimators <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                        'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))
        labels = ['Weight sharing', 'Basis expansion model']
        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=map_generator,
                    sampler=testing_sampler,
                    estimator=current_estimator)
            # Plot results
        print(RMSE)

        G2 = GFigure(xaxis=np.rint(1024 * sampling_factor),
                     yaxis=RMSE[0, :],
                     xlabel='Number of measurements, ' + r'$\vert \Omega \vert $',
                     ylabel="RMSE(dB)",
                     legend=labels[0])
        if n_run_estimators > 1:
            for ind_plot in range(n_run_estimators - 1):
                G2.add_curve(xaxis=np.rint(1024 * sampling_factor), yaxis=RMSE[ind_plot + 1, :],
                             legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        G.append(G2)
        return G

    #  RMSE comparison with Gudmundson maps: 64 X 64 grid with 1.5 m spacing vs  32 X 32 grid with 3 m spacing
    def experiment_1021(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(400)

        # Generator
        v_central_freq = [1.4e9]
        num_precomputed_sh = 600000

        # Sampler
        sampling_factor = np.concatenate((np.linspace(0.05, 0.1, 10, endpoint=False), np.linspace(0.1, 0.2, 7)),
                                         axis=0)[0:14]
        testing_sampler = MapSampler()

        # Estimators
        v_architecture_ids = ['8b', '8']
        v_grid_size = [64, 32]
        v_filters = [27, 27]
        v_code_length = [256, 64]

        v_num_maps = [num_precomputed_sh, 500000]
        ve_split_frac = 1
        v_epochs = [100, 100]
        v_learning_rate = [5e-4, 5e-4]
        v_superbatches = [4, 1]
        v_sampling_factor = [0.05, 0.2]
        v_sampling_diff_rat = [4, 1]  # to have the same number of measurements for both grids

        # 1. Generators and autoencoder estimators
        labels = ["64 X 64 grid (1.5 m spacing)", "32 X 32 grid (3 m spacing)"]
        all_map_generators = []
        all_estimators = []

        for ind_est in range(len(v_architecture_ids)):

            # Generator
            map_generator = GudmundsonMapGenerator(
                n_grid_points_x=v_grid_size[ind_est],
                n_grid_points_y=v_grid_size[ind_est],
                # tx_power=np.tile(np.array([[11, 7], [10, 6]]), (int(np.size(v_central_freq) / 2), 1)), # dBm
                tx_power_interval=[5, 11],  # dBm
                b_shadowing=True,
                num_precomputed_shadowing_mats=num_precomputed_sh,
                v_central_frequencies=v_central_freq)
            all_map_generators += [map_generator]

            # autoencoder estimator
            if ind_est == 0:
                estimator = AutoEncoderEstimator(
                    n_pts_x=map_generator.n_grid_points_x,
                    n_pts_y=map_generator.n_grid_points_y,
                    arch_id=v_architecture_ids[ind_est],
                    c_length=v_code_length[ind_est],
                    bases_vals=map_generator.m_basis_functions,
                    n_filters=v_filters[ind_est],
                    weight_file=
                    'output/autoencoder_experiments/savedWeights/weights_trained_est1.h5')

                # Train autoencoder
                training_sampler = MapSampler(v_sampling_factor=[v_sampling_factor[0] / v_sampling_diff_rat[ind_est],
                                                                 v_sampling_factor[1] / v_sampling_diff_rat[ind_est]])
                history, codes = estimator.train(generator=map_generator,
                                                 sampler=training_sampler,
                                                 learning_rate=v_learning_rate[ind_est],
                                                 n_super_batches=v_superbatches[ind_est],
                                                 n_maps=v_num_maps[ind_est],
                                                 perc_train=0.9,
                                                 v_split_frac=ve_split_frac,
                                                 n_resamples_per_map=1,
                                                 n_epochs=v_epochs[ind_est])

                # Plot training results: losses and visualize codes if enabled
                ExperimentSet.plot_histograms_of_codes_and_visualize(
                    map_generator.x_length,
                    map_generator.y_length,
                    codes,
                    estimator.chosen_model,
                    exp_num,
                )
                ExperimentSet.plot_train_and_val_losses(history, exp_num)
            else:
                estimator = AutoEncoderEstimator(
                    n_pts_x=map_generator.n_grid_points_x,
                    n_pts_y=map_generator.n_grid_points_y,
                    arch_id=v_architecture_ids[ind_est],
                    c_length=v_code_length[ind_est],
                    bases_vals=map_generator.m_basis_functions,
                    n_filters=v_filters[ind_est],
                    weight_file=
                    'output/autoencoder_experiments/savedWeights/weights_trained_est2.h5'
                )

            estimator.str_name = labels[ind_est]
            all_estimators += [estimator]


        # Simulation pararameters
        n_runs = 500

        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        estimators_to_sim = [1, 2]
        assert len(estimators_to_sim) <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                              'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))

        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                if ind_est == 0:
                    testing_sampler.v_sampling_factor = sampling_factor[ind_sampling] / 4
                else:
                    testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=all_map_generators[estimators_to_sim[ind_est] -
                                                 1],
                    sampler=testing_sampler,
                    estimator=current_estimator)

        # Plot results
        print(RMSE)
        G = GFigure(
            xaxis=np.rint(
                all_map_generators[1].n_grid_points_x * all_map_generators[1].n_grid_points_y * sampling_factor),
            yaxis=RMSE[0, :],
            xlabel="Number of measurements, " + r"$\vert \Omega \vert $",
            ylabel="RMSE(dB)",
            legend=labels[0])
        if len(estimators_to_sim) >= 1:
            for ind_plot in range(len(estimators_to_sim) - 1):
                G.add_curve(xaxis=np.rint(
                    all_map_generators[1].n_grid_points_x * all_map_generators[
                        1].n_grid_points_y * sampling_factor),
                    yaxis=RMSE[ind_plot + 1, :], legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        return G

    #  RMSE comparison with the Wireless Insite maps: 64 X 64 grid with 1.5 m spacing vs  32 X 32 grid with 3 m spacing
    def experiment_1022(self):
        # Execution parameters
        exp_num = int(
            re.search(r'\d+',
                      sys._getframe().f_code.co_name).group())
        # np.random.seed(400)

        # Sampler
        sampling_factor = np.concatenate((np.linspace(0.05, 0.1, 10, endpoint=False), np.linspace(0.1, 0.2, 7)),
                                         axis=0)
        testing_sampler = MapSampler()

        # Estimators
        v_architecture_ids = ['8b', '8']
        v_grid_size = [64, 32]
        v_filters = [27, 27]
        v_code_length = [256, 64]

        v_num_maps = [125, 125000]
        ve_split_frac = [0.5, 0.5]
        v_epochs = [2, 100]
        v_superbatches = [1, 1]
        v_sampling_factor = [0.05, 0.2]
        v_sampling_diff_rat = [4, 1]  # to have the same number of measurements for both grids

        # 1. Generators and autoencoder estimators
        labels = ["64 X 64 grid (1.5 m spacing)", "32 X 32 grid (3 m spacing)"]
        all_map_generators = []
        all_estimators = []

        for ind_est in range(len(v_architecture_ids)):

            if v_grid_size[ind_est] == 32:
                distance_fac = 2
            else:
                distance_fac = 1

            # Generator
            testing_generator = InsiteMapGenerator(
                n_grid_points_x=v_grid_size[ind_est],
                n_grid_points_y=v_grid_size[ind_est],
                l_file_num=np.arange(41, 43),
                inter_grid_points_dist_factor=distance_fac)
            all_map_generators += [testing_generator]

            # autoencoder estimator
            if ind_est == 2:
                estimator = AutoEncoderEstimator(
                    n_pts_x=testing_generator.n_grid_points_x,
                    n_pts_y=testing_generator.n_grid_points_y,
                    arch_id=v_architecture_ids[ind_est],
                    c_length=v_code_length[ind_est],
                    bases_vals=testing_generator.m_basis_functions,
                    n_filters=v_filters[ind_est],
                    weight_file=
                    'output/autoencoder_experiments/savedWeights/weights_trained_est1.h5')

            else:
                estimator = AutoEncoderEstimator(
                    n_pts_x=testing_generator.n_grid_points_x,
                    n_pts_y=testing_generator.n_grid_points_y,
                    arch_id=v_architecture_ids[ind_est],
                    c_length=v_code_length[ind_est],
                    bases_vals=testing_generator.m_basis_functions,
                    n_filters=v_filters[ind_est])
                    # weight_file=
                    # 'output/autoencoder_experiments/savedWeights/weights_trained_est2.h5')

            # Train autoencoder
            training_generator = InsiteMapGenerator(
                n_grid_points_x=v_grid_size[ind_est],
                n_grid_points_y=v_grid_size[ind_est],
                l_file_num=np.arange(1, 41),
                inter_grid_points_dist_factor=distance_fac)
            training_sampler = MapSampler(v_sampling_factor=[v_sampling_factor[0] / v_sampling_diff_rat[ind_est],
                                                             v_sampling_factor[1] / v_sampling_diff_rat[ind_est]])
            history, codes = estimator.train(generator=training_generator,
                                             sampler=training_sampler,
                                             learning_rate=5e-4,
                                             n_super_batches=v_superbatches[ind_est],
                                             n_maps=v_num_maps[ind_est],
                                             perc_train=0.9,
                                             v_split_frac=ve_split_frac,
                                             n_resamples_per_map=10,
                                             n_epochs=v_epochs[ind_est])

            ExperimentSet.plot_train_and_val_losses(history, exp_num)

            estimator.str_name = labels[ind_est]
            all_estimators += [estimator]

        # Simulation pararameters
        n_runs = 1000

        simulator = Simulator(n_runs=n_runs, use_parallel_proc=False)

        # run the simulation
        estimators_to_sim = [2]
        assert len(estimators_to_sim) <= len(all_estimators), 'The number of estimators to simulate must be ' \
                                                              'less or equal to the total number of estimators'
        RMSE = np.zeros((len(estimators_to_sim), len(sampling_factor)))

        for ind_est in range(len(estimators_to_sim)):
            current_estimator = all_estimators[estimators_to_sim[ind_est] -
                                               1]
            for ind_sampling in range(np.size(sampling_factor)):
                if ind_est == 0:
                    testing_sampler.v_sampling_factor = sampling_factor[ind_sampling] / 4
                else:
                    testing_sampler.v_sampling_factor = sampling_factor[ind_sampling]
                RMSE[ind_est, ind_sampling] = simulator.simulate(
                    generator=all_map_generators[estimators_to_sim[ind_est] -
                                                 1],
                    sampler=testing_sampler,
                    estimator=current_estimator)

        # Plot results
        print(RMSE)
        G = GFigure(
            xaxis=np.rint(
                970 * sampling_factor),
            yaxis=RMSE[0, :],
            xlabel="Number of measurements, " + r"$\vert \Omega \vert $",
            ylabel="RMSE(dB)",
            legend=labels[0])
        if len(estimators_to_sim) >= 1:
            for ind_plot in range(len(estimators_to_sim) - 1):
                G.add_curve(xaxis=np.rint(
                    970 * sampling_factor),
                    yaxis=RMSE[ind_plot + 1, :], legend=labels[ind_plot + 1])
        ExperimentSet.plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num, labels)
        return G


    @staticmethod
    def plot_reconstruction(x_len,
                            y_len,
                            l_true_map,
                            l_sampled_maps,
                            l_masks,
                            realization_sampl_fac,
                            meta_data,
                            l_reconstructed_maps,
                            exp_num):
        # sim_real_maps=False):
        # Computes and prints the error
        vec_meta = meta_data.flatten()
        vec_map = l_true_map[0].flatten()
        vec_est_map = l_reconstructed_maps[0].flatten()
        err = np.sqrt((npla.norm((1 - vec_meta) * (vec_map - vec_est_map))) ** 2 / len(np.where(vec_meta == 0)[0]))
        print('The realization error is %.5f' % err)

        # Set plot_in_db  to True if the map entries are in natural units and have to be displayed in dB

        for in_truemap in range(len(l_true_map)):
            for ind_1 in range(l_true_map[in_truemap].shape[0]):
                for ind_2 in range(l_true_map[in_truemap].shape[1]):
                    if meta_data[ind_1][ind_2] == 1:
                        l_true_map[in_truemap][ind_1][ind_2] = 'NaN'

        for in_reconsmap in range(len(l_reconstructed_maps)):
            for ind_1 in range(l_reconstructed_maps[in_reconsmap].shape[0]):
                for ind_2 in range(l_reconstructed_maps[in_reconsmap].shape[1]):
                    if meta_data[ind_1][ind_2] == 1:
                        l_reconstructed_maps[in_reconsmap][ind_1][ind_2] = 'NaN'

        # tr_map = ax1.imshow(db_to_dbm(l_true_map[0]),
        #                     interpolation='bilinear',
        #                     extent=(0, x_len, 0, y_len),
        #                     cmap='jet',
        #                     origin='lower',
        #                     vmin=v_min,
        #                     vmax=v_max)  #

        for in_samplmap in range(len(l_sampled_maps)):
            for ind_1 in range(l_sampled_maps[in_samplmap].shape[0]):
                for ind_2 in range(l_sampled_maps[in_samplmap].shape[1]):
                    if l_masks[in_samplmap][ind_1][ind_2] == 0 or meta_data[ind_1][ind_2] == 1:
                        l_sampled_maps[in_samplmap][ind_1][ind_2] = 'NaN'

        fig1 = plt.figure(figsize=(15, 5))
        fig1.subplots_adjust(hspace=0.7, wspace=0.4)
        n_rows = len(l_true_map)
        n_cols = len(l_sampled_maps) + len(l_reconstructed_maps) + 1
        v_min = -100
        v_max = -40
        tr_im_col = []
        for ind_row in range(n_rows):
            ax = fig1.add_subplot(n_rows, n_cols, 1)
            im_tr = ax.imshow(db_to_dbm(l_true_map[ind_row][:, :]),
                           interpolation='bilinear',
                           extent=(0, x_len, 0, y_len),
                           cmap='jet',
                           origin='lower',
                           vmin=v_min,
                           vmax=v_max)
            tr_im_col = im_tr
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title('True map')

        for ind_col in range(len(l_sampled_maps)):
            ax = fig1.add_subplot(n_rows, n_cols, ind_col + 2)
            im = ax.imshow(db_to_dbm(l_sampled_maps[ind_col][:, :]),
                           extent=(0, x_len, 0, y_len),
                           cmap='jet',
                           origin='lower',
                           vmin=v_min,
                           vmax=v_max)
            ax.set_xlabel('x [m]')
            ax.set_title('Sampled map \n' + r'$\vert \Omega \vert $=%d' % (np.rint(len(np.where(vec_meta == 0)[0]) *
                                                                               realization_sampl_fac[0])))

        for ind_col in range(len(l_reconstructed_maps)):
            ax = fig1.add_subplot(n_rows, n_cols, ind_col + len(l_sampled_maps) + 2)
            im = ax.imshow(db_to_dbm(l_reconstructed_maps[ind_col][:, :]),
                           interpolation='bilinear',
                           extent=(0, x_len, 0, y_len),
                           cmap='jet',
                           origin='lower',
                           vmin=v_min,
                           vmax=v_max)
            ax.set_xlabel('x [m]')
            ax.set_title('Reconstructed map \n' + r'$\vert \Omega \vert $=%d' % (np.rint(len(np.where(vec_meta == 0)[0]) *
                                                                               realization_sampl_fac[ind_col])))

        fig1.subplots_adjust(right=0.85)
        cbar_ax = fig1.add_axes([0.88, 0.28, 0.02, 0.43])
        fig1.colorbar(tr_im_col, cax=cbar_ax, label='dBm')

        # plt.show()  # (block=False)
        # plt.pause(10)
        fig1.savefig(
            'output/autoencoder_experiments/savedResults/True_Sampled_and_Rec_maps%d.pdf'
            % exp_num)
        return

    @staticmethod
    def plot_b_coefficients(x_len,
                            y_len,
                            coeffs,
                            bases_to_plot,
                            labels,
                            exp_num,
                            file_name):
        # Set plot_in_db  to True if the map entries are in natural units and have to be displayed in dB
        fig1 = plt.figure()
        fig1.subplots_adjust(hspace=0.7, wspace=0.4)
        v_min = -120
        v_max = -60
        n_bases_to_plot = np.size(bases_to_plot)
        n_rows = coeffs.shape[0]
        for ind_base in range(n_bases_to_plot):
            for ind_row in range(n_rows):
                ax = fig1.add_subplot(n_rows, n_bases_to_plot, ind_base + 1 + ind_row * n_bases_to_plot)
                im = ax.imshow(coeffs[ind_row, :, :, bases_to_plot[ind_base]],
                               interpolation='bilinear',
                               extent=(0, x_len, 0, y_len),
                               cmap='jet',
                               origin='lower',
                               vmin=v_min,
                               vmax=v_max)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_title(labels[ind_row] + '\n $\pi_{%d}(\mathbf{x})$' % (bases_to_plot[ind_base] + 1))
        fig1.subplots_adjust(right=0.85)
        cbar_ax = fig1.add_axes([0.85, 0.11, 0.03, 0.77])
        fig1.colorbar(im, cax=cbar_ax, label='dB')

        # plt.show()
        # plt.pause(10)
        with open(
                'output/autoencoder_experiments/savedResults/' + str(file_name) + '_%d.pickle'
                % exp_num, 'wb') as f_tsrec:
            pickle.dump(coeffs,
                        f_tsrec)
        fig1.savefig(
            'output/autoencoder_experiments/savedResults/' + str(file_name) + '_%d.pdf'
            % exp_num)
        return

    @staticmethod
    def plot_and_save_RMSE_vs_sf(sampling_factor, RMSE, exp_num, labels):
        with open(
                'output/autoencoder_experiments/savedResults/RMSE_%d.pickle' %
                exp_num, 'wb') as f_RMSE:
            pickle.dump(RMSE, f_RMSE)
        fig = plt.figure()
        clrs_list = ['b', 'r', 'g', 'k']  # list of basic colors
        styl_list = ['-', '--', '-.', ':']  # list of basic linestyles
        makers_list = ['*', 's', '>', 'v']
        n_curves = RMSE.shape[0]
        for ind_curv in range(n_curves):
            if n_curves == 1:
                RMSE_plot = RMSE
                label = labels
            else:
                RMSE_plot = RMSE[ind_curv, :]
                label = labels[ind_curv]
            plt.plot(sampling_factor,
                     RMSE_plot.T,
                     linestyle=styl_list[ind_curv],
                     marker=makers_list[ind_curv],
                     color=clrs_list[ind_curv],
                     label=label)
        plt.legend()
        plt.xlabel("Number of layers")
        plt.ylabel('RMSE(dB)')
        plt.grid()
        # plt.show()
        fig.savefig(
            'output/autoencoder_experiments/savedResults/RMSE_vrs_SF_%d.pdf' %
            exp_num)

    @staticmethod
    def plot_and_save_RMSE_vs_sf_modified(sampling_factor, RMSE, exp_num,
                                          labels):
        with open(
                'output/autoencoder_experiments/savedResults/RMSE_%d.pickle' %
                exp_num, 'wb') as f_RMSE:
            pickle.dump(RMSE, f_RMSE)
        fig = plt.figure()
        clrs_list = ['b', 'r', 'g', 'm', 'k', 'c']  # list of basic colors
        styl_list = ['-', '-', '-.', '--', ':', '--']  # list of basic linestyles
        makers_list = ['d', '>', 's', '*', 'o', 'v']
        n_curves = RMSE.shape[0]
        for ind_curv in range(n_curves):
            RMSE_plot = RMSE[ind_curv, :]
            label = labels[ind_curv]
            plt.plot(np.rint(970 * sampling_factor), # multiply by 1024 instead of 970  for Gudmundson dataset
                     RMSE_plot.T,
                     linestyle=styl_list[ind_curv],
                     marker=makers_list[ind_curv],
                     color=clrs_list[ind_curv],
                     label=label)
        plt.legend()
        plt.xlabel('Number of measurements, ' + r'$\vert \Omega \vert $')
        plt.ylabel('RMSE (dB)')
        plt.grid()
        # plt.show()
        fig.savefig(
            'output/autoencoder_experiments/savedResults/RMSE_vrs_SF_%d.pdf' %
            exp_num)

    @staticmethod
    def plot_and_save_RMSE_vs_code(code_vals, RMSE, exp_num, label):
        fig = plt.figure()
        plt.plot(code_vals,
                 RMSE.T,
                 linestyle='-.',
                 marker='v',
                 color='b',
                 label=label)
        plt.legend()
        plt.xlabel('Code length, $N_\lambda$')
        plt.ylabel('RMSE(dB)')
        plt.grid()
        with open(
                'output/autoencoder_experiments/savedResults/RMSE_vrs_code_%d.pickle'
                % exp_num, 'wb') as f_RMSE:
            pickle.dump(RMSE, f_RMSE)
        # plt.show()
        fig.savefig(
            'output/autoencoder_experiments/savedResults/RMSE_vrs_code_%d.pdf'
            % exp_num)

    @staticmethod
    def plot_train_and_val_losses(history, exp_num):
        fig = plt.figure()
        plt.plot(history[0], color='b')
        plt.plot(history[1], color='r')
        # plt.plot(history[2, :], marker='>')
        plt.grid()
        plt.title('model loss')
        plt.ylabel('loss(MSE)')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'],
                   loc='upper right')  # , 'train_internal'
        with open(
                'output/autoencoder_experiments/savedResults/Tr_and_Val_loss_%d.pickle'
                % exp_num, 'wb') as f_trval:
            pickle.dump(history, f_trval)
        # plt.show()
        fig.savefig(
            'output/autoencoder_experiments/savedResults/Tr_and_Val_loss_%d.pdf'
            % exp_num)

    @staticmethod
    def plot_histograms_of_codes_and_visualize(
            x_len,
            y_len,
            codes,
            model,
            exp_num,
            visualize=False,
            use_cov_matr=True):
        # Plotting and saving code histograms
        with open(
                'output/autoencoder_experiments/savedResults/codes_%d.pickle' %
                exp_num, 'wb') as f_codes:
            pickle.dump(codes, f_codes)

        last_layer_name = model.get_layer('encoder').layers[- 1].name
        if last_layer_name == 'latent_dense':
            resh_codes = codes
        else:
            resh_codes = np.reshape(codes, (codes.shape[0], codes.shape[1] *
                                            codes.shape[2] * codes.shape[3]))

        num_latents_var = resh_codes.shape[1]
        if num_latents_var <= 2:
            fig1 = plt.figure()
            fig1.subplots_adjust(hspace=0.4, wspace=0.4)
            for ind_lat_var in range(num_latents_var):
                assert num_latents_var % 2 == 0, 'Set the number of latent variables to an even number'
                ax = fig1.add_subplot(int(num_latents_var / 2), 2,
                                      ind_lat_var + 1)
                ax.hist(resh_codes[:, ind_lat_var], bins=1000)
                ax.set_title('hist($[\lambda]_%d)$' % (ind_lat_var + 1))
                # ax.set_title( r'hist($[\boldsymbol{\lambda}]_%d)$' % (ind_lat_var + 1))
            fig1.savefig(
                'output/autoencoder_experiments/savedResults/hist_%d.pdf' %
                exp_num)

        # Visualization
        if visualize:
            matplotlib.rcParams.update({'font.size': 8})
            trained_decoder = model.get_layer('decoder')
            out_layer_ind = len(trained_decoder.layers) - 1
            # decoder_out_ind = np.random.randint(num_codes_to_show)
            # avg_code = np.mean(codes, axis=0)
            avg_code = codes[np.random.randint(codes.shape[0]), :]

            fig2 = plt.figure(constrained_layout=True, figsize=(6, 13))
            main_gs = fig2.add_gridspec(4, 1)
            fig2.subplots_adjust(hspace=0.8, wspace=0.4)
            decoder_output = get_layer_activations(
                trained_decoder, np.expand_dims(avg_code, axis=0),
                out_layer_ind)
            ax = fig2.add_subplot(main_gs[0, :])
            ax.imshow(
                db_to_dbm(np.reshape(
                            decoder_output[0, :, 0],
                            (model.input_shape[1], model.input_shape[2]))),
                extent=(0, x_len, 0, y_len),
                cmap='jet',
                origin='lower',
                vmin=-90,
                vmax=-20)
            if use_cov_matr:
                ax.set_title(r'$ \mathbf{\lambda}=\dot \mathbf{\lambda}$')
            else:
                ax.set_title(r'$ \mathbf{\lambda}= \mathbf{\lambda}_{avg}$')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

            if use_cov_matr:
                n_viz_maps_chang = 6
                for ind_viz in range(n_viz_maps_chang):
                    if last_layer_name == 'latent_dense':
                        resh_codes = codes
                    else:
                        resh_codes = np.reshape(
                            codes,
                            (codes.shape[0],
                             codes.shape[1] * codes.shape[2] * codes.shape[3]),
                            order='F')
                    cov_code = np.cov(resh_codes, rowvar=False)
                    eig_vals_cov, eig_vecs_cov = np.linalg.eig(cov_code)
                    if ind_viz > 2:
                        ind_viz_jump = ind_viz + 40  # the jump depends on the code length
                    else:
                        ind_viz_jump = ind_viz

                    input_codes = avg_code.flatten('F') + 10 * eig_vecs_cov[:, ind_viz_jump]
                    if last_layer_name == 'latent_dense':
                        resh_input_codes = input_codes
                    else:
                        resh_input_codes = np.reshape(
                            input_codes,
                            (codes.shape[1], codes.shape[2], codes.shape[3]),
                            order='F')
                    decoder_output = get_layer_activations(
                        trained_decoder,
                        np.expand_dims(resh_input_codes, axis=0),
                        out_layer_ind)
                    ax = fig2.add_subplot(4, 2, ind_viz + 3)
                    ax.imshow(
                        db_to_dbm(np.reshape(
                            decoder_output[0, :, 0],
                            (model.input_shape[1], model.input_shape[2]))),
                        extent=(0, x_len, 0, y_len),  # 100 is the area side
                        cmap='jet',
                        origin='lower',
                        vmin=-90,
                        vmax=-20)
                    axis_title = r'$ \mathbf{\lambda}=\dot \mathbf{\lambda} + \alpha \mathbf{v}_{%d}$' % (
                        ind_viz_jump + 1)
                    ax.set_title(axis_title)
                    ax.set_xlabel('x [m]')
                    ax.set_ylabel('y [m]')
            else:
                std_code = np.std(codes, axis=0).flatten('F')
                all_combos = list(
                    itertools.combinations(np.arange(np.size(avg_code)), 2))
                n_comb = len(all_combos)
                for ind_comb in range(n_comb):
                    current_comb = np.array(all_combos[ind_comb])
                    input_codes = avg_code.flatten('F')
                    input_codes[current_comb] = input_codes[
                                                    current_comb] - std_code[current_comb]
                    if last_layer_name == 'latent_dense':
                        resh_input_codes = input_codes
                    else:
                        resh_input_codes = np.reshape(
                            input_codes,
                            (codes.shape[1], codes.shape[2], codes.shape[3]),
                            order='F')
                    decoder_output = get_layer_activations(
                        trained_decoder,
                        np.expand_dims(resh_input_codes, axis=0),
                        out_layer_ind)
                    ax = fig2.add_subplot(int(n_comb / 2 + 1), 2, ind_comb + 3)
                    ax.imshow(
                        db_to_dbm(np.reshape(
                            decoder_output[0, :, 0],
                            (model.input_shape[1], model.input_shape[2]))),
                        extent=(0, x_len, 0, y_len),  # 100 is the area side
                        cmap='jet',
                        origin='lower',
                        vmin=-90,
                        vmax=-20)
                    ax.set_title(r'$\mathcal{S}=\{%d, %d\}$' %
                        (current_comb[0] + 1, current_comb[1] + 1))
                    ax.set_xlabel('x [m]')
                    ax.set_ylabel('y [m]')
            fig2.savefig(
                'output/autoencoder_experiments/savedResults/Visualiz_%d.pdf' %
                exp_num)
        if use_cov_matr:
            plt.show()

        
def plot_training_coeficients(generator, autoencoder, exp_num, file_name='True_and_Training_bcoeffs'):
    true_l_maps_train, sampled_maps_train = pickle.load(
        open("output/autoencoder_experiments/savedResults/True_and_Est_training_bcoeffs.pickle", "rb"))
    rand_map_ind = np.random.choice(true_l_maps_train.shape[0])
    sampled_map_feed = np.expand_dims(sampled_maps_train[rand_map_ind, :, :, :], axis=0)

    estimated_coeffs = autoencoder.get_autoencoder_coefficients_dec(sampled_map_feed)
    l_bem_maps = [true_l_maps_train[rand_map_ind, :, :, :], estimated_coeffs[0]]
    bases_to_plot = np.arange(l_bem_maps[0].shape[2])
    reconst_labels = ['True', 'Training coefficients']

    ExperimentSet.plot_b_coefficients(generator.x_length,
                                      generator.y_length,
                                      np.array(l_bem_maps),
                                      bases_to_plot,
                                      reconst_labels,
                                      exp_num,
                                      file_name)
