import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow.keras.utils as ku
from utils.communications import db_to_natural
import numpy as np
import sys


class Autoencoder:
    def __init__(self,
                 height,
                 width,
                 c_len,
                 add_mask_channel,
                 mask_as_tensor,
                 bases,
                 n_filters=32,
                 activ_function_name=None,
                 kernel_size=(3, 3),
                 conv_stride=1,
                 pool_size_str=2,
                 use_batch_norm=False):
        self.height = height
        self.width = width
        self.add_mask_channel = add_mask_channel
        self.mask_as_tensor = mask_as_tensor
        self.bases = bases
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.pool_size_str = pool_size_str
        self.use_batch_norm = use_batch_norm
        self.c_len = c_len
        self.norm_hereToo = False

        if activ_function_name:
            self.activ_func = 'obtain_%s' % activ_function_name
        else:
            self.activ_func = 'obtain_leakyrelu'

        if self.activ_func == 'obtain_prelu':
            self.activ_func_dense = 'obtain_prelu_for_dense_lay'
        else:
            self.activ_func_dense = self.activ_func


     
    # 10 layer network: 5 for encoder and 5 for decoder
    def convolutional_autoencoder_1(self):

        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        x = keras.layers.Conv2D(filters=self.n_filters,
                                kernel_size=self.kernel_size,
                                strides=self.conv_stride,
                                kernel_initializer='he_normal',
                                activation=getattr(Autoencoder, self.activ_func)(),
                                padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        x = keras.layers.Conv2D(filters=self.n_filters,
                                kernel_size=self.kernel_size,
                                strides=self.conv_stride,
                                kernel_initializer='he_normal',
                                activation=getattr(Autoencoder, self.activ_func)(),
                                padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')

        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        x = keras.layers.Conv2DTranspose(filters=self.n_filters,
                                         kernel_size=self.kernel_size,
                                         strides=self.conv_stride,
                                         kernel_initializer='he_normal',
                                         activation=getattr(Autoencoder, self.activ_func)(),
                                         padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        x = keras.layers.Conv2DTranspose(filters=self.bases.shape[1],
                                         kernel_size=self.kernel_size,
                                         strides=self.conv_stride,
                                         kernel_initializer='he_normal',
                                         activation=getattr(Autoencoder, self.activ_func)(),
                                         padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 14 layer network: 7 for encoder and 7 for  decoder
    def convolutional_autoencoder_2(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks

        n_layers_and_n_filters = [1, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters, self.n_filters]

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[1]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 18 layer network: 9 for encoder and 9 for decoder
    def convolutional_autoencoder_3(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters, self.n_filters]

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = int(self.n_filters)
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        x = keras.layers.Conv2D(filters=self.n_filters,
                                kernel_size=self.kernel_size,
                                strides=self.conv_stride,
                                kernel_initializer='he_normal',
                                activation=getattr(Autoencoder, self.activ_func)(),
                                padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        x = keras.layers.Conv2DTranspose(filters=self.n_filters,
                                         kernel_size=self.kernel_size,
                                         strides=self.conv_stride,
                                         kernel_initializer='he_normal',
                                         activation=getattr(Autoencoder, self.activ_func)(),
                                         padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[1]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 22 layer network: 11 for encoder and 11 for decoder (architecture close to the adopted in the conference paper)
    def convolutional_autoencoder_4(self):

        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 2
        n_layers_and_n_filters3 = [self.n_filters] * 3

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters3:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters3[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[1]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 26 layer network: 13 for encoder and 13 for decoder
    def convolutional_autoencoder_5(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 3

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[0]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 30 layer network: 15 for encoder and 15 for decoder
    def convolutional_autoencoder_6(self):

        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 3
        n_layers_and_n_filters3 = [self.n_filters] * 2

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters3:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters3:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters3[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters3[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[1]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 34 layer network: 17 for encoder and 17 for decoder
    def convolutional_autoencoder_7(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 3

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[1]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 26 layer network: 13 for encoder and 13 for decoder (fully convolutional) , code length 4
    def convolutional_autoencoder_8toy(self):
            # Build the Autoencoder Model

            if self.add_mask_channel:
                if self.mask_as_tensor:
                    n_channels = self.bases.shape[1] + 2
                else:
                    n_channels = self.bases.shape[1] + 1
            else:
                n_channels = self.bases.shape[1]

            # First build the Encoder Model
            inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
            x = inputs

            # Stacking  Conv2D, Activations, and AveragePooling layers blocks
            n_layers_and_n_filters = [1, self.n_filters, self.n_filters]
            n_layers_and_n_filters2 = [self.n_filters] * 3

            for filters in n_layers_and_n_filters:
                if filters == 1:
                    filters = self.n_filters
                if self.height == 64:  # Assume square inputs
                    kernel_to_use = (5, 5)
                else:
                    kernel_to_use = self.kernel_size
                x = keras.layers.Conv2D(filters=filters,
                                        kernel_size=kernel_to_use,
                                        strides=self.conv_stride,
                                        kernel_initializer='he_normal',
                                        activation=keras.layers.LeakyReLU(alpha=0.3),
                                        padding='same')(x)
                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

            x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                              strides=self.pool_size_str,
                                              padding='same')(x)

            for filters in n_layers_and_n_filters2:
                x = keras.layers.Conv2D(filters=filters,
                                        kernel_size=self.kernel_size,
                                        strides=self.conv_stride,
                                        kernel_initializer='he_normal',
                                        activation=keras.layers.LeakyReLU(alpha=0.3),
                                        padding='same')(x)
                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

            x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                              strides=self.pool_size_str,
                                              padding='same')(x)

            for filters in n_layers_and_n_filters2:
                x = keras.layers.Conv2D(filters=filters,
                                        kernel_size=self.kernel_size,
                                        strides=self.conv_stride,
                                        kernel_initializer='he_normal',
                                        activation=keras.layers.LeakyReLU(alpha=0.3),
                                        padding='same')(x)
                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

            x = keras.layers.AveragePooling2D(pool_size=2 * self.pool_size_str,
                                              strides=2 * self.pool_size_str,
                                              padding='same')(x)

            shape_here = keras.backend.int_shape(x)

            x = keras.layers.Conv2D(filters=int(self.c_len / (shape_here[1] * shape_here[2])),
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

            # Shape information to use in the decoder
            latent = x
            shape = keras.backend.int_shape(latent)

            # Instantiate Encoder Model
            encoder = Model(inputs, latent, name='encoder')
            encoder.summary()

            # Build the Decoder Model
            latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
            x = latent_inputs

            # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
            x = keras.layers.Conv2DTranspose(filters=int(self.c_len / (shape_here[1] * shape_here[2])),
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

            x = keras.layers.UpSampling2D(size=2 * self.pool_size_str,
                                          interpolation='bilinear')(x)

            for filters in n_layers_and_n_filters2[::-1]:
                x = keras.layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=self.kernel_size,
                                                 strides=self.conv_stride,
                                                 kernel_initializer='he_normal',
                                                 activation=keras.layers.LeakyReLU(alpha=0.3),
                                                 padding='same')(x)
                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

            x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                          interpolation='bilinear')(x)

            for filters in n_layers_and_n_filters2[::-1]:
                x = keras.layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=self.kernel_size,
                                                 strides=self.conv_stride,
                                                 kernel_initializer='he_normal',
                                                 activation=keras.layers.LeakyReLU(alpha=0.3),
                                                 padding='same')(x)
                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

            x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                          interpolation='bilinear')(x)

            for filters in n_layers_and_n_filters[::-1]:
                if filters == 1:
                    filters = self.bases.shape[0]
                if self.height == 64:  # Assume square inputs
                    kernel_to_use = (5, 5)
                else:
                    kernel_to_use = self.kernel_size
                x = keras.layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_to_use,
                                                 strides=self.conv_stride,
                                                 kernel_initializer='he_normal',
                                                 activation=keras.layers.LeakyReLU(alpha=0.3),
                                                 padding='same')(x)
                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
            x = tf.cast(x, tf.float64)
            outputs = 10 * (tf.keras.backend.log(
                tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
                tf.cast(10, tf.float64)))
            outputs = tf.reshape(outputs, [-1] + [
                outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

            # Instantiate Decoder Model
            decoder = Model(latent_inputs, outputs, name='decoder')
            decoder.summary()

            # Instantiate Autoencoder Model
            autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
            autoencoder.summary()

            # Plot the Autoencoder Model
            # plot_the_model(encoder, decoder, autoencoder)
            return autoencoder

    # 26 layer network: 13 for encoder and 13 for decoder (fully convolutional)
    def convolutional_autoencoder_8(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            if self.mask_as_tensor:
                n_channels = self.bases.shape[1] + 2
            else:
                n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 3

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            if self.height == 64:    # Assume square inputs
                kernel_to_use = (5, 5)
            else:
                kernel_to_use = self.kernel_size
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=kernel_to_use,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        shape_here = keras.backend.int_shape(x)

        x = keras.layers.Conv2D(filters=int(self.c_len / (shape_here[1] * shape_here[2])),
                                kernel_size=self.kernel_size,
                                strides=self.conv_stride,
                                kernel_initializer='he_normal',
                                activation=keras.layers.LeakyReLU(alpha=0.3),
                                padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        # Shape information to use in the decoder
        latent = x
        shape = keras.backend.int_shape(latent)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
        x = latent_inputs

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.Conv2DTranspose(filters=int(self.c_len / (shape_here[1] * shape_here[2])),
                                         kernel_size=self.kernel_size,
                                         strides=self.conv_stride,
                                         kernel_initializer='he_normal',
                                         activation=keras.layers.LeakyReLU(alpha=0.3),
                                         padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[0]
            if self.height == 64:    # Assume square inputs
                kernel_to_use = (5, 5)
            else:
                kernel_to_use = self.kernel_size
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=kernel_to_use,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 32 layer network: 13 for encoder and 13 for decoder (fully convolutional)
    def convolutional_autoencoder_9(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            if self.mask_as_tensor:
                n_channels = self.bases.shape[1] + 2
            else:
                n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 4

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            if self.height == 64:  # Assume square inputs
                kernel_to_use = (5, 5)
            else:
                kernel_to_use = self.kernel_size
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=kernel_to_use,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        shape_here = keras.backend.int_shape(x)

        x = keras.layers.Conv2D(filters=int(self.c_len / (shape_here[1] * shape_here[2])),
                                kernel_size=self.kernel_size,
                                strides=self.conv_stride,
                                kernel_initializer='he_normal',
                                activation=keras.layers.LeakyReLU(alpha=0.3),
                                padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        # Shape information to use in the decoder
        latent = x
        shape = keras.backend.int_shape(latent)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
        x = latent_inputs

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.Conv2DTranspose(filters=int(self.c_len / (shape_here[1] * shape_here[2])),
                                         kernel_size=self.kernel_size,
                                         strides=self.conv_stride,
                                         kernel_initializer='he_normal',
                                         activation=keras.layers.LeakyReLU(alpha=0.3),
                                         padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[0]
            if self.height == 64:  # Assume square inputs
                kernel_to_use = (5, 5)
            else:
                kernel_to_use = self.kernel_size
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=kernel_to_use,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
        x = tf.cast(x, tf.float64)
        outputs = 10 * (tf.keras.backend.log(
            tf.matmul(db_to_natural(x), tf.cast(self.bases, tf.float64))) / tf.keras.backend.log(
            tf.cast(10, tf.float64)))
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # OLD: 32 layer network: 16 for encoder and 16 for decoder (fully convolutional)
    def convolutional_autoencoder_9_old(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 4

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=keras.layers.LeakyReLU(alpha=0.3),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        shape_here = keras.backend.int_shape(x)

        x = keras.layers.Conv2D(filters=int(self.c_len / (shape_here[1] * shape_here[2])),
                                kernel_size=self.kernel_size,
                                strides=self.conv_stride,
                                kernel_initializer='he_normal',
                                activation=keras.layers.LeakyReLU(alpha=0.3),
                                padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        # Shape information to use in the decoder
        latent = x
        shape = keras.backend.int_shape(latent)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
        x = latent_inputs

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.Conv2DTranspose(filters=4,
                                         kernel_size=self.kernel_size,
                                         strides=self.conv_stride,
                                         kernel_initializer='he_normal',
                                         activation=keras.layers.LeakyReLU(alpha=0.3),
                                         padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[0]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=keras.layers.LeakyReLU(alpha=0.3),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
        x = tf.cast(x, tf.float64)
        outputs = tf.matmul(x, self.bases)
        outputs = tf.reshape(outputs, [-1] + [
            outputs.shape[1] * outputs.shape[2] * outputs.shape[3], 1])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    # 40 layer network: 20 for encoder and 20 for decoder
    def convolutional_autoencoder_10(self):
        # Build the Autoencoder Model

        if self.add_mask_channel:
            n_channels = self.bases.shape[1] + 1
        else:
            n_channels = self.bases.shape[1]

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
        x = inputs

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 4

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=self.conv_stride,
                                    kernel_initializer='he_normal',
                                    activation=getattr(Autoencoder, self.activ_func)(),
                                    padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        # Shape information to use in the decoder
        shape = keras.backend.int_shape(x)

        # Generate the latent vector
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dropout(rate=0.1)(x)
        latent = keras.layers.Dense(self.c_len,
                                    activation=getattr(Autoencoder, self.activ_func_dense)(),
                                    name='latent_dense')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
        x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
                               activation=getattr(Autoencoder, self.activ_func_dense)())(latent_inputs)
        # x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = self.bases.shape[1]
            x = keras.layers.Conv2DTranspose(filters=filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.conv_stride,
                                             kernel_initializer='he_normal',
                                             activation=getattr(Autoencoder, self.activ_func)(),
                                             padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        outputs = x

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder
    

    @staticmethod
    def obtain_leakyrelu():
        leakrelu = keras.layers.LeakyReLU(alpha=0.3)
        return leakrelu

    @staticmethod
    def obtain_prelu():
        prelu = keras.layers.PReLU(shared_axes=[1, 2])
        return prelu

    @staticmethod
    def obtain_prelu_for_dense_lay():
        prelu = keras.layers.PReLU()
        return prelu


def plot_the_model(encoder, decoder, autoencoder):
    ku.plot_model(encoder,
                  to_file='Models/encoder_model.pdf',
                  show_shapes=True,
                  show_layer_names=True)
    ku.plot_model(decoder,
                  to_file='Models/decoder_model.pdf',
                  show_shapes=True,
                  show_layer_names=True)
    ku.plot_model(autoencoder,
                  to_file='Models/autoencoder_model.pdf',
                  show_shapes=True,
                  show_layer_names=True)


# ==============================
# class OldArchitectures:
#     # 28 layer network fully convolutional: more kernels when approaching the bottleneck
#     # added the average pooling and upsampling layers close to the bottleneck in the 32  by 32 maps to have the same
#     # code height as in the 16 by 16 maps.
#     def convolutional_autoencoder_2(self):
#
#         # Build the Autoencoder Model
#
#         if self.add_mask_channel:
#             n_channels = 2
#         else:
#             n_channels = 1
#
#         # First build the Encoder Model
#         inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
#         x = inputs
#
#         # Stacking  Conv2D, Activations, and AveragePooling layers blocks
#         n_layers_and_n_filters = [1, int(self.n_filters / 8)]
#         n_layers_and_n_filters2 = [int(self.n_filters / 4)] * 3
#         n_layers_and_n_filters3 = [int(self.n_filters / 2)] * 3
#         n_layers_and_n_filters4 = list(np.concatenate(([int(self.n_filters)] * 2, [1]), axis=0))
#
#         for filters in n_layers_and_n_filters:
#             if filters == 1:
#                 filters = int(self.n_filters / 8)
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=self.activ_func,
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=self.activ_func,
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters3:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=self.activ_func,
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters4:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=self.activ_func,
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#         latent = x
#         shape = keras.backend.int_shape(latent)
#
#         # Instantiate Encoder Model
#         encoder = Model(inputs, latent, name='encoder')
#         encoder.summary()
#
#         # Build the Decoder Model
#         latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
#         x = latent_inputs
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters4[::-1]:
#             if filters == 1:
#                 filters = self.n_filters
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=self.activ_func,
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters3[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=self.activ_func,
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=self.activ_func,
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=self.activ_func,
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         outputs = x
#
#         # Instantiate Decoder Model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#         decoder.summary()
#
#         # Instantiate Autoencoder Model
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
#         autoencoder.summary()
#
#         # Plot the Autoencoder Model
#         # plot_the_model(encoder, decoder, autoencoder)
#
#         return autoencoder
#
#     # 28 layer network (6 dense layers, more kernels when approaching the bottleneck):
#     def convolutional_autoencoder_3(self):
#
#         # Build the Autoencoder Model
#
#         if self.add_mask_channel:
#             n_channels = 2
#         else:
#             n_channels = 1
#
#         # First build the Encoder Model
#         inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
#         x = inputs
#
#         # Stacking  Conv2D, Activations, and AveragePooling layers blocks
#         n_layers_and_n_filters = [1, int(self.n_filters / 8)]
#         n_layers_and_n_filters2 = [int(self.n_filters / 4)] * 3
#         n_layers_and_n_filters3 = [int(self.n_filters / 2)] * 3
#
#         for filters in n_layers_and_n_filters:
#             if filters == 1:
#                 filters = int(self.n_filters / 8)
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters3:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         # Shape information to use in the decoder
#         shape = keras.backend.int_shape(x)
#
#         # Generate the latent vector
#         x = keras.layers.Flatten()(x)
#         # x = keras.layers.Dropout(rate=0.1)(x)
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         latent = keras.layers.Dense(self.c_len,
#                                     activation=keras.layers.PReLU(),
#                                     name='latent_dense')(x)
#
#         # Instantiate Encoder Model
#         encoder = Model(inputs, latent, name='encoder')
#         encoder.summary()
#
#         # Build the Decoder Model
#         latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(latent_inputs)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#         # x = keras.layers.Dropout(rate=0.1)(x)
#         x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)
#
#         # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters3[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         outputs = x
#
#         # Instantiate Decoder Model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#         decoder.summary()
#
#         # Instantiate Autoencoder Model
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
#         autoencoder.summary()
#
#         # Plot the Autoencoder Model
#         # plot_the_model(encoder, decoder, autoencoder)
#         return autoencoder
#
#     # 28 layer network fully convolutional: fewer kernels when approaching the bottleneck
#     # added the average pooling and upsampling layers close to the bottleneck in the 32  by 32 maps to have the same
#     # code height as in the 16 by 16 maps.
#     def convolutional_autoencoder_2b(self):
#
#         # Build the Autoencoder Model
#
#         if self.add_mask_channel:
#             n_channels = 2
#         else:
#             n_channels = 1
#
#         # First build the Encoder Model
#         inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
#         x = inputs
#
#         # Stacking  Conv2D, Activations, and AveragePooling layers blocks
#         n_layers_and_n_filters = [1, int(self.n_filters)]
#         n_layers_and_n_filters2 = [int(self.n_filters / 2)] * 3
#         n_layers_and_n_filters3 = [int(self.n_filters / 4)] * 3
#         n_layers_and_n_filters4 = list(np.concatenate(([int(self.n_filters / 8)] * 2, [1]), axis=0))
#
#         for filters in n_layers_and_n_filters:
#             if filters == 1:
#                 filters = int(self.n_filters)
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters3:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters4:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#         latent = x
#         shape = keras.backend.int_shape(latent)
#
#         # Instantiate Encoder Model
#         encoder = Model(inputs, latent, name='encoder')
#         encoder.summary()
#
#         # Build the Decoder Model
#         latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
#         x = latent_inputs
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters4[::-1]:
#             if filters == 1:
#                 filters = int(self.n_filters / 8)
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters3[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         outputs = x
#
#         # Instantiate Decoder Model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#         decoder.summary()
#
#         # Instantiate Autoencoder Model
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
#         autoencoder.summary()
#
#         # Plot the Autoencoder Model
#         # plot_the_model(encoder, decoder, autoencoder)
#
#         return autoencoder
#
#     # 28 layer network (6 dense layers, fewer kernels when approaching the bottleneck):
#     def convolutional_autoencoder_3b(self):
#
#         # Build the Autoencoder Model
#
#         if self.add_mask_channel:
#             n_channels = 2
#         else:
#             n_channels = 1
#
#         # First build the Encoder Model
#         inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
#         x = inputs
#
#         # Stacking  Conv2D, Activations, and AveragePooling layers blocks
#         n_layers_and_n_filters = [1, int(self.n_filters)]
#         n_layers_and_n_filters2 = [int(self.n_filters / 2)] * 3
#         n_layers_and_n_filters3 = [int(self.n_filters / 4)] * 3
#
#         for filters in n_layers_and_n_filters:
#             if filters == 1:
#                 filters = int(self.n_filters)
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters3:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         # Shape information to use in the decoder
#         shape = keras.backend.int_shape(x)
#
#         # Generate the latent vector
#         x = keras.layers.Flatten()(x)
#         # x = keras.layers.Dropout(rate=0.1)(x)
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         latent = keras.layers.Dense(self.c_len,
#                                     activation=keras.layers.PReLU(),
#                                     name='latent_dense')(x)
#
#         # Instantiate Encoder Model
#         encoder = Model(inputs, latent, name='encoder')
#         encoder.summary()
#
#         # Build the Decoder Model
#         latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(latent_inputs)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#         # x = keras.layers.Dropout(rate=0.1)(x)
#         x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)
#
#         # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters3[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         outputs = x
#
#         # Instantiate Decoder Model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#         decoder.summary()
#
#         # Instantiate Autoencoder Model
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
#         autoencoder.summary()
#
#         # Plot the Autoencoder Model
#         # plot_the_model(encoder, decoder, autoencoder)
#         return autoencoder
#
#     # 28 layer network fully convolutional : same number of kernels on all conv. and convTr. layers
#     # added the average pooling and upsampling layers close to the bottleneck in the 32  by 32 maps to have the same
#     # code height as in the 16 by 16 maps.
#     def convolutional_autoencoder_4(self):
#
#         # Build the Autoencoder Model
#
#         if self.add_mask_channel:
#             n_channels = 2
#         else:
#             n_channels = 1
#
#         # First build the Encoder Model
#         inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
#         x = inputs
#
#         # Stacking  Conv2D, Activations, and AveragePooling layers blocks
#         n_layers_and_n_filters = [np.size(self.bases), int(self.n_filters)]
#         n_layers_and_n_filters2 = [self.n_filters] * 3
#         n_layers_and_n_filters3 = list(np.concatenate(([self.n_filters] * 2, [1]), axis=0))
#
#         for filters in n_layers_and_n_filters:
#             if filters == np.size(self.bases):
#                 filters = int(self.n_filters)
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters3:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#         latent = x
#         shape = keras.backend.int_shape(latent)
#
#         # Instantiate Encoder Model
#         encoder = Model(inputs, latent, name='encoder')
#         encoder.summary()
#
#         # Build the Decoder Model
#         latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
#         x = latent_inputs
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#         for filters in n_layers_and_n_filters3[::-1]:
#             if filters == 1:
#                 filters = self.n_filters
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         # base_vals = np.random.normal(loc=0, size=self.num_bases)
#         # h = tf.scan(lambda a, outputs: tf.multiply(outputs, base_vals), x) tf.transpose(x, [0, 3, 1, 2]
#         if np.size(self.bases) == 1:
#             outputs = x
#         else:
#             outputs = tf.reduce_sum(x * self.bases[0], axis=3)
#             outputs = tf.expand_dims(outputs, axis=3)
#         # Instantiate Decoder Model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#         decoder.summary()
#
#         # Instantiate Autoencoder Model
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
#         autoencoder.summary()
#
#         # Plot the Autoencoder Model
#         # plot_the_model(encoder, decoder, autoencoder)
#
#         return autoencoder
#
#         # 22 layer network : more convolutional layers as one approaches the bottleneck
#
#     # 28 layer network (6 dense layers):
#     def convolutional_autoencoder_5(self):
#
#         # Build the Autoencoder Model
#
#         if self.add_mask_channel:
#             n_channels = 2
#         else:
#             n_channels = 1
#
#         # First build the Encoder Model
#         inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
#         x = inputs
#
#         # Stacking  Conv2D, Activations, and AveragePooling layers blocks
#         n_layers_and_n_filters = [1, int(self.n_filters)]
#         n_layers_and_n_filters2 = [int(self.n_filters), int(self.n_filters), int(self.n_filters)]
#
#         for filters in n_layers_and_n_filters:
#             if filters == 1:
#                 filters = int(self.n_filters)
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         # Shape information to use in the decoder
#         shape = keras.backend.int_shape(x)
#
#         # Generate the latent vector
#         x = keras.layers.Flatten()(x)
#         # x = keras.layers.Dropout(rate=0.1)(x)
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         latent = keras.layers.Dense(self.c_len,
#                                     activation=keras.layers.PReLU(),
#                                     name='latent_dense')(x)
#
#         # Instantiate Encoder Model
#         encoder = Model(inputs, latent, name='encoder')
#         encoder.summary()
#
#         # Build the Decoder Model
#         latent_inputs = keras.layers.Input(shape=(self.c_len,), name='decoder_input')
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(latent_inputs)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#
#         x = keras.layers.Dense(shape[1] * shape[2] * shape[3],
#                                activation=keras.layers.PReLU())(x)
#         # x = keras.layers.Dropout(rate=0.1)(x)
#         x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)
#
#         # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         outputs = x
#
#         # Instantiate Decoder Model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#         decoder.summary()
#
#         # Instantiate Autoencoder Model
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
#         autoencoder.summary()
#
#         # Plot the Autoencoder Model
#         # plot_the_model(encoder, decoder, autoencoder)
#         return autoencoder
#
#     # 32 layer network fully convolutional : same number of kernels on all conv. and convTr. layers
#     # added the average pooling and upsampling layers close to the bottleneck in the 32  by 32 maps to have the same
#     # code height as in the 16 by 16 maps.
#     def convolutional_autoencoder_6(self):
#
#         # Build the Autoencoder Model
#
#         if self.add_mask_channel:
#             n_channels = 2
#         else:
#             n_channels = 1
#
#         # First build the Encoder Model
#         inputs = keras.layers.Input(shape=(self.height, self.width, n_channels), name='encoder_input')
#         x = inputs
#
#         # Stacking  Conv2D, Activations, and AveragePooling layers blocks
#         n_layers_and_n_filters = [1, int(self.n_filters)]
#         n_layers_and_n_filters2 = [self.n_filters] * 3
#         n_layers_and_n_filters3 = [self.n_filters] * 4
#         n_layers_and_n_filters4 = list(np.concatenate(([self.n_filters] * 3, [1]), axis=0))
#
#         for filters in n_layers_and_n_filters:
#             if filters == 1:
#                 filters = int(self.n_filters)
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters2:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters3:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         for filters in n_layers_and_n_filters4:
#             x = keras.layers.Conv2D(filters=filters,
#                                     kernel_size=self.kernel_size,
#                                     strides=self.conv_stride,
#                                     kernel_initializer='he_normal',
#                                     activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                     padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
#                                           strides=self.pool_size_str,
#                                           padding='same')(x)
#
#         latent = x
#         shape = keras.backend.int_shape(latent)
#
#         # Instantiate Encoder Model
#         encoder = Model(inputs, latent, name='encoder')
#         encoder.summary()
#
#         # Build the Decoder Model
#         latent_inputs = keras.layers.Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')
#         x = latent_inputs
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters4[::-1]:
#             if filters == 1:
#                 filters = self.n_filters
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters3[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters2[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         x = keras.layers.UpSampling2D(size=self.pool_size_str,
#                                       interpolation='bilinear')(x)
#
#         for filters in n_layers_and_n_filters[::-1]:
#             x = keras.layers.Conv2DTranspose(filters=filters,
#                                              kernel_size=self.kernel_size,
#                                              strides=self.conv_stride,
#                                              kernel_initializer='he_normal',
#                                              activation=keras.layers.PReLU(shared_axes=[1, 2]),
#                                              padding='same')(x)
#             if self.use_batch_norm:
#                 x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
#
#         outputs = x
#
#         # Instantiate Decoder Model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#         decoder.summary()
#
#         # Instantiate Autoencoder Model
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
#         autoencoder.summary()
#
#         # Plot the Autoencoder Model
#         # plot_the_model(encoder, decoder, autoencoder)
#
#         return autoencoder
#
#         # 22 layer network : more convolutional layers as one approaches the bottleneck
