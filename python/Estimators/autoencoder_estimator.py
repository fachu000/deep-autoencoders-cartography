import time
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np
from Estimators.map_estimator import BemMapEstimator
from Models.net_models import Autoencoder
from joblib import Parallel, delayed
import multiprocessing
import pickle
import pdb
import os
import scipy.io


class AutoEncoderEstimator(BemMapEstimator):
    """
       Arguments:
           n_grid_points_x : number of grid points along x-axis
           n_grid_points_y : number of grid points along y-axis
           add_mask_channel : flag for adding the mask as the second channel at the input of the autoencoder: set to
           False by default.
           transfer_learning : flag for disabling the transfer learning, by default it is set to True.
           learning_rate:  learning rate
       """

    def __init__(self,
                 n_pts_x=32,
                 n_pts_y=32,
                 arch_id='8',
                 c_length=4,
                 n_filters=32,
                 activ_func_name=None,
                 add_mask_channel=True,
                 use_masks_as_tensor=False,
                 weight_file=None,
                 **kwargs):
        super(AutoEncoderEstimator, self).__init__(**kwargs)
        self.str_name = "Proposed"
        self.n_grid_points_x = n_pts_x
        self.n_grid_points_y = n_pts_y
        self.code_length = c_length
        self.add_mask_channel = add_mask_channel
        self.use_masks_as_tensor = use_masks_as_tensor
        self.n_filters = n_filters
        self.activ_func_name = activ_func_name
        architecture_name = 'convolutional_autoencoder_%s' % arch_id
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        self.chosen_model = getattr(Autoencoder(height=self.n_grid_points_x,
                                                width=self.n_grid_points_y,
                                                c_len=self.code_length,
                                                add_mask_channel=self.add_mask_channel,
                                                mask_as_tensor=self.use_masks_as_tensor,
                                                bases=self.bases_vals,
                                                n_filters=self.n_filters,
                                                activ_function_name=self.activ_func_name), architecture_name)()
        self.chosen_model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                                  loss='mse',
                                  sample_weight_mode='temporal')
        if weight_file:
            self.chosen_model.load_weights(weight_file)
        # self.chosen_model.summary()
        # quit()

    def estimate_map(self, sampled_map, mask, meta_data):
        """

        :param sampled_map:  the sampled map with incomplete entries, 2D array with shape (n_grid_points_x,
                                                                                           n_grid_points_y)
        :type sampled_map: float
        :param mask: is a binary array of the same size as the sampled map:
        :return: the reconstructed map,  2D array with the same shape as the sampled map
        """
        if self.add_mask_channel:
            sampled_map_exp = np.expand_dims(sampled_map, axis=0)
            mask_exp = np.expand_dims(np.expand_dims(mask, axis=0), axis=3)
            meta_exp = np.expand_dims(np.expand_dims(meta_data, axis=0), axis=3)
            if self.use_masks_as_tensor:
                # add the masks as a tensor
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp, -meta_exp), axis=3)
            else:
                # combine masks into a single matrix
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp - meta_exp), axis=3)
        else:
            sampled_map_feed = np.expand_dims(sampled_map, axis=0)
        reconstructed = self.chosen_model.predict(x=sampled_map_feed)

        return np.reshape(reconstructed[0, :, :], sampled_map.shape)

    def estimate_bem_coefficient_map(self, sampled_map, mask, meta_data):

        # obtain coefficients from the autoencoder estimator
        if self.add_mask_channel:
            sampled_map_exp = np.expand_dims(sampled_map, axis=0)
            mask_exp = np.expand_dims(np.expand_dims(mask, axis=0), axis=3)
            meta_exp = np.expand_dims(np.expand_dims(meta_data, axis=0), axis=3)
            if self.use_masks_as_tensor:
                # add the masks as a tensor
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp, -meta_exp), axis=3)
            else:
                # combine masks into a single matrix
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp - meta_exp), axis=3)
        else:
            sampled_map_feed = np.expand_dims(sampled_map, axis=0)
        estimated_coeffs = self.get_autoencoder_coefficients_dec(sampled_map_feed)
        return estimated_coeffs[0]

    def train(self,
              generator,
              sampler,
              learning_rate=1e-5,
              n_super_batches=1,
              **kwargs):
        hist = []
        latent_vars = []
        for ind_s_batch in range(n_super_batches):
            l_rate = learning_rate / (2 ** ind_s_batch)
            hist, latent_vars = self.train_one_batch(generator, sampler, l_rate, **kwargs)

        return hist, latent_vars

    def train_one_batch(self,
                        generator,
                        sampler,
                        l_rate,
                        n_maps=128,
                        perc_train=0.9,
                        v_split_frac=1,
                        n_resamples_per_map=1,
                        n_epochs=10,
                        batch_size=64,
                        enable_noisy_targets=False,
                        l_fraction_maps_from_each_sampler=None):

        """
        ARGUMENTS:
        `generator`: object descending from class MapGenerator
        `sampler` : object of class Sampler or list of objects of class Sampler.
        `v_split_frac` : if 1, then targets are the entire maps. If tuple or list of length 2, then two splits of the
        sampled maps are used as input and target. The input contains a fraction v_split_frac[0]  of the samples
        whereas the target contains a fraction v_split_frac[1]  of the samples.
        `l_fraction_maps_from_each_sampler` : list of float between 0 and 1 that add up to 1. The n-th entry indicates
        the fraction of maps to be sampled with `sampler[n]`. If set to None, an equal number of maps are
        sampled with each sampler in `sampler`.   [FUTURE, OPTIONAL]
        """

        start_time = time.time()

        # function to generate training points
        def process_one_map(ind_map):
            """
            Retuns a list of data points  where one data point is a dictionary
            with keys _point", "y_point", "y_mask", and "channel_power_map"

            """
            t_map, m_meta_map, t_ch_power = generator.generate()
            t_sampled_map, m_mask = sampler.sample_map(t_map, m_meta_map)
            if v_split_frac == 1:

                # m_mask_and_meta = m_mask - m_meta_map
                m_mask_out = 1 - m_meta_map

                # reshaping and adding masks
                t_sampled_map_in, v_map_out, v_mask_out = self.format_preparation(t_sampled_map, m_meta_map, m_mask,
                                                                                  t_map, m_mask_out)

                data_point = {"x_point": t_sampled_map_in,  # Nx X Ny X Nf (+1)
                              "y_point": v_map_out,  # Nx Ny Nf
                              "y_mask": v_mask_out,  # Nx Ny Nf
                              "channel_power_map": t_ch_power}  # Nx X Ny X B
                l_data_points = [data_point]

            elif len(v_split_frac) == 2:
                l_data_points = []
                for ind_resample in range(n_resamples_per_map):
                    # resample the sampled map
                    t_sampled_map_in, m_mask_in, t_sampled_map_out, m_mask_out = sampler.resample_map(
                        t_sampled_map, m_mask, v_split_frac)

                    # reshaping and adding masks
                    t_sampled_map_in, v_map_out, v_mask_out = self.format_preparation(t_sampled_map_in, m_meta_map, m_mask_in,
                                                                                      t_sampled_map_out, m_mask_out)

                    data_point = {"x_point": t_sampled_map_in,  # Nx X Ny X Nf (+1)
                                  "y_point": v_map_out,  # Nx Ny Nf
                                  "y_mask": v_mask_out,  # Nx Ny Nf
                                  "channel_power_map": t_ch_power}  # Nx X Ny X B
                    l_data_points.append(data_point)

            else:
                Exception("invalid value of v_split_frac")
                l_data_points = None
            return l_data_points

        # Generate all data points using parallel processing
        num_cores = int(multiprocessing.cpu_count() / multiprocessing.cpu_count())
        l_l_data_points = Parallel(n_jobs=num_cores, backend='threading')(delayed(process_one_map)(ind_map)
                                                                          for ind_map in range(int(n_maps)))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('The elapsed time for generating and sampling training and validation maps using multiple processors '
              'with %d cores is' % num_cores, time.strftime("%d:%H:%M:%S", time.gmtime(elapsed_time)))

        # pdb.set_trace()

        # Get arrays from l_l_data_points
        def ll_to_nparray(str_key):
            return np.array(
                [data_point[str_key] for l_data_points in l_l_data_points for data_point in l_data_points])
        t_x_points = ll_to_nparray("x_point")
        t_y_points = ll_to_nparray("y_point")
        t_y_masks = ll_to_nparray("y_mask")
        t_channel_pows = ll_to_nparray("channel_power_map")

        # Training/validation split
        overall_n_maps = t_x_points.shape[0]
        n_maps_train = int(perc_train * overall_n_maps)

        # x_points
        t_x_points_train = t_x_points[0:n_maps_train]
        t_x_points_valid = t_x_points[n_maps_train:]

        # y_points
        t_y_points_train = t_y_points[0:n_maps_train]
        t_y_points_valid = t_y_points[n_maps_train:]

        # y_masks
        t_y_masks_train = t_y_masks[0:n_maps_train]
        t_y_masks_valid = t_y_masks[n_maps_train:]

        # training channel powers
        t_channel_pows_train = t_channel_pows[0:n_maps_train]

        # Training loss computation using callback
        internal_eval = InternalEvaluation(
            validation_data=(t_x_points_train, t_y_points_train, t_y_masks_train))

        # Fit
        self.chosen_model.optimizer.optimizer._lr = l_rate
        train_history = self.chosen_model.fit(x=t_x_points_train,
                                              y=t_y_points_train,
                                              batch_size=batch_size,
                                              sample_weight=t_y_masks_train,
                                              epochs=n_epochs,
                                              validation_data=(
                                                  t_x_points_valid, t_y_points_valid, t_y_masks_valid),
                                              callbacks=[internal_eval],
                                              verbose=2)
        history = np.array([np.array(internal_eval.train_losses),
                            np.array(train_history.history['val_loss']),
                            np.array(train_history.history['loss'])])

        # save the weights
        self.chosen_model.save_weights(
            'output/autoencoder_experiments/savedWeights/weights.h5')  # To Do: remove 1026 exp number

        # obtain the codes for some training maps
        trained_encoder = self.chosen_model.get_layer('encoder')
        # trained_encoder.summary()
        # trained_decoder.summary()
        out_layer_ind = len(trained_encoder.layers) - 1
        num_codes_to_show = min(2 * n_maps_train, int(3e3))
        codes = get_layer_activations(trained_encoder, t_x_points_train[0: num_codes_to_show], out_layer_ind)

        # save some l_training_coefficients for later checking
        n_saved_maps = 500
        with open(
                'output/autoencoder_experiments/savedResults/True_and_Est_training_bcoeffs.pickle'
                , 'wb') as f_bcoeff:
            pickle.dump([t_channel_pows_train[0:n_saved_maps], t_x_points_train[0:n_saved_maps]],
                        f_bcoeff)
        return history, codes

    def format_preparation(self,
                           t_input,
                           meta_data,
                           m_mask_in,
                           t_output,
                           m_mask_out,
                           ):
        """
        Returns:
        `t_input_proc`: Nx x Ny x Nf (+1) tensor
        `v_output`: Nx Ny Nf vector (vectorized Nx x Ny x Nf tensor).
        `v_weight_out`: vector of the same dimension as `v_output`.

        """

        if self.add_mask_channel:
            m_mask_exp = np.expand_dims(m_mask_in, axis=2)
            m_meta_exp = np.expand_dims(meta_data, axis=2)
            if self.use_masks_as_tensor:
                # add the masks as a tensor
                t_input_proc = np.concatenate((t_input, m_mask_exp, - m_meta_exp), axis=2)
            else:
                # combine masks into a single matrix
                t_input_proc = np.concatenate((t_input, m_mask_exp - m_meta_exp), axis=2)
        else:
            t_input_proc = t_input

        v_output = np.expand_dims(np.ndarray.flatten(t_output), axis=1)
        t_mask_out = np.repeat(m_mask_out[:, :, np.newaxis], t_output.shape[2], axis=2)
        v_weight_out = np.ndarray.flatten(t_mask_out)

        return t_input_proc, v_output, v_weight_out

    def get_autoencoder_coefficients_dec(self,  t_input):
        encoder = self.chosen_model.get_layer('encoder')
        decoder = self.chosen_model.get_layer('decoder')
        enc_out_layer_ind = len(encoder.layers) - 1
        ind_layer_dec = len(decoder.layers) - 9  # subtract 9 layers to reach the layer giving the coefficients
        codes = get_layer_activations(encoder, t_input, enc_out_layer_ind)
        coeff = get_layer_activations(decoder, codes, ind_layer_dec)
        return coeff  # [:, :, :, 0:(self.bases_vals.shape[0] - 1)] -1 to remove the noise base if it is included




class InternalEvaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.X_data, self.Y_data, self.weights = validation_data

    def on_train_begin(self, logs={}):
        self.train_losses = []

    def on_epoch_end(self, epoch, logs={}):
        train_loss = self.model.evaluate(x=self.X_data, y=self.Y_data, sample_weight=self.weights, batch_size=64,
                                         verbose=1)
        self.train_losses.append(train_loss)
        print("Internal evaluation - epoch: {:d} - loss: {:.6f}".format(epoch, train_loss))
        return self.train_losses


def get_layer_activations(network, m_input, ind_layer):
    f_get_layer_activations = K.function([network.layers[0].input],
                                         [network.layers[ind_layer].output])
    layer_activations = f_get_layer_activations(m_input)[0]
    return layer_activations



