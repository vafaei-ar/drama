from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pylab as plt
from tensorflow.keras.layers import Input, Flatten, Reshape, Dense, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Cropping2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

from .nn_utils import *

def encoder_cnn_2d(x,
                   filters,
                   kernel_size,
                   strides,
                   activation,
                   pool_size,
                   activity_regularizer):
    
    y = Conv2D(filters=filters, 
               kernel_size=kernel_size,
               strides=strides,
               activation=activation,
               padding='same',
               activity_regularizer=activity_regularizer)(x)
    
    if pool_size and pool_size!=(1,1):
        y = MaxPooling2D(pool_size=pool_size,
                         strides=None,
                         padding='same')(y)
    return y

def decoder_cnn_2d(x,
                   filters,
                   kernel_size,
                   strides,
                   activation,
                   upsample_size,
                   activity_regularizer,
                   transpose = False):
    
    if transpose:
        y = Conv2DTranspose(filters = filters,
                            kernel_size = kernel_size,
                            strides = strides,
                            padding='same',
                            dilation_rate = (1, 1),
                            activation = activation,
                            activity_regularizer = activity_regularizer)(x)
    else:
        y = Conv2D(filters = filters, 
                   kernel_size = kernel_size,
                   strides = strides,
                   activation = activation,
                   padding = 'same',
                   activity_regularizer = activity_regularizer)(x)
    
    if upsample_size and upsample_size!=(1,1):
        y = UpSampling2D(size=upsample_size)(y)
        
        

        
    return y

def cond2d_arch_encoder(x_in,
                        latent_dim,
                        n_conv,
                        activation = 'relu',
                        filters = 4,
                        filter_factor = 2,
                        last_filters = None,
                        kernel_size = (5,5),
                        strides = (1,1),
                        pool_size = (2,2),
                        activity_regularizer = None):

    x = x_in

    for _ in range(n_conv-1):
        x = encoder_cnn_2d(x,
                          filters = filters,
                          kernel_size = kernel_size,
                          strides = strides,
                          activation = activation,
                          pool_size = pool_size,
                          activity_regularizer = activity_regularizer)
        filters = int(filters*filter_factor)
        
    if last_filters is None:
        last_filters = filters

    x = encoder_cnn_2d(x,
                       filters = last_filters,
                       kernel_size = kernel_size,
                       strides = strides,
                       activation = activation,
                       pool_size = 0,
                       activity_regularizer = activity_regularizer)

    x_shape = x.get_shape().as_list()

    x = Flatten()(x)

    encoded = Dense(latent_dim,
              activation=activation)(x)
    
    return encoded,x_shape

def cond2d_arch_decoder(encoded,
                        decoded_dim_x,
                        decoded_dim_y,
                        n_conv,
                        x_shape,
                        activation = 'relu',
                        filters = 4,
                        filter_factor = 2,
                        last_filters = 1,
                        kernel_size = (5,5),
                        strides = (1,1),
                        upsample_size = (2,2),
                        activity_regularizer = None,
                        transpose = False):

    x = Dense(x_shape[1]*x_shape[2]*x_shape[3],
              activation=activation)(encoded)

    x = Reshape(x_shape[1:])(x)

    for _ in range(n_conv-1):
        x = decoder_cnn_2d(x,
                           filters = filters,
                           kernel_size = kernel_size,
                           strides = strides,
                           activation = activation,
                           upsample_size = upsample_size,
                           activity_regularizer = activity_regularizer,
                           transpose = transpose)
        filters = int(filters//filter_factor)
        
    decoded = decoder_cnn_2d(x,
                             filters = last_filters,
                             kernel_size = kernel_size,
                             strides = (1,1),
                             activation = activation,
                             upsample_size = 0,
                             activity_regularizer = activity_regularizer,
                             transpose = transpose)
    
    d_shape = decoded.get_shape().as_list()
    delta_x = d_shape[1] - decoded_dim_x
    delta_y = d_shape[2] - decoded_dim_y
    
    assert (delta_x>=0 and delta_y>=0),'Dimension error! The architecture input-output is not compatible!'

    d_crop_x = int(delta_x/2)
    d_crop_y = int(delta_y/2)

    if delta_x!=0 or delta_y!=0:
        
        if delta_x%2==0:
            d_crop_x = (d_crop_x,d_crop_x)
        else:
            d_crop_x = (d_crop_x,d_crop_x+1)
            
        if delta_y%2==0:
            d_crop_y = (d_crop_y,d_crop_y)
        else:
            d_crop_y = (d_crop_y,d_crop_y+1)
    
        decoded = Cropping2D(cropping=(d_crop_x,d_crop_y))(decoded)
    
    return decoded


class ConvolutionalAutoEncoder2D(AutoEncoderBase):
    def __init__(self,
                 input_dim_x,
                 input_dim_y,
                 latent_dim,
                 n_conv = 2,
                 filters = 4,
                 filter_factor = 2,
                 kernel_size = (5,5),
                 strides_en = (1,1),
                 strides_de = (2,2),
                 pool_size = (2,2),
                 upsample_size = False,
                 activation = 'relu',
                 transpose = True,
                 activity_regularizer = None,
                 learning_rate = 0.03,
                 batch_size = 100):
        
        super().__init__(input_dim = None,
                         latent_dim = latent_dim,
                         activation = activation,
                         activity_regularizer = activity_regularizer,
                         learning_rate = learning_rate,
                         batch_size = batch_size)
        
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.input_dim = input_dim_x*input_dim_y
        self.n_conv = n_conv
        self.filters = filters
        self.filter_factor = filter_factor
        self.kernel_size = kernel_size
        self.strides_en = strides_en
        self.strides_de = strides_de
        self.pool_size = pool_size
        self.upsample_size = upsample_size
        self.transpose = transpose
        self.activity_regularizer = activity_regularizer
        
        self.input = Input(shape=(self.input_dim_x, self.input_dim_y, 1)) 
        self.encoded = self._encode(self.input)
        self.decoded = self._decode(self.encoded)
        self.encoder_index = len(Model(self.input,self.encoded).layers)
#         self.encoder_index = 12
        self.prepare_models()
        
    def _encode(self, x_in):

        encoded,self.x_shape = cond2d_arch_encoder(x_in,
                                                   self.latent_dim,
                                                   self.n_conv,
                                                   activation = self.activation,
                                                   filters = self.filters,
                                                   filter_factor = self.filter_factor,
                                                   kernel_size = self.kernel_size,
                                                   strides = self.strides_en,
                                                   pool_size = self.pool_size,
                                                   activity_regularizer = self.activity_regularizer)

        return encoded
    
    def _decode(self, encoded):

        decoded = cond2d_arch_decoder(encoded,
                                      self.input_dim_x,
                                      self.input_dim_y,
                                      self.n_conv,
                                      self.x_shape,
                                      activation = self.activation,
                                      filters = self.x_shape[-1],
                                      filter_factor = self.filter_factor,
                                      kernel_size = self.kernel_size,
                                      strides = self.strides_de,
                                      upsample_size = self.upsample_size,
                                      activity_regularizer = self.activity_regularizer,
                                      transpose = self.transpose)
        
        return decoded

    def train(self, x, 
              training_epochs=10, 
              lr_sch=True, 
              early_stop=True,
              verbose=True): 
        if x.ndim==3:
            x = np.expand_dims(x,-1)
        super().train(x, 
                      training_epochs=training_epochs, 
                      lr_sch=lr_sch, 
                      early_stop=early_stop,
                      verbose=verbose)
        
    def encoder(self, x):
        if x.ndim==3:
            x = np.expand_dims(x,-1)
        assert (x.shape[1]==self.input_dim_x and x.shape[2]==self.input_dim_y),'Input dimension problem!'
        return self._encoder.predict(x)


class ConvolutionalVariationalAutoEncoder2D(AutoEncoderBase):
    def __init__(self,
                 input_dim_x,
                 input_dim_y,
                 latent_dim,
                 n_conv = 2,
                 filters = 4,
                 filter_factor = 2,
                 kernel_size = (5,5),
                 strides_en = (1,1),
                 strides_de = (2,2),
                 pool_size = (2,2),
                 upsample_size = False,
                 activation = 'relu',
                 transpose = True,
                 activity_regularizer = None,
                 learning_rate = 0.03,
                 batch_size = 100):
        
        super().__init__(input_dim = None,
                         latent_dim = latent_dim,
                         activation = activation,
                         activity_regularizer = activity_regularizer,
                         learning_rate = learning_rate,
                         batch_size = batch_size,
                         variational = True)
        
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.input_dim = input_dim_x*input_dim_y
        self.n_conv = n_conv
        self.filters = filters
        self.filter_factor = filter_factor
        self.kernel_size = kernel_size
        self.strides_en = strides_en
        self.strides_de = strides_de
        self.pool_size = pool_size
        self.upsample_size = upsample_size
        self.transpose = transpose
        self.activity_regularizer = activity_regularizer
        
        self.input = Input(shape=(self.input_dim_x, self.input_dim_y, 1)) 
        self.encoded,self.z_mean, self.z_log_var = self._encode(self.input)
        self.decoded = self._decode(self.encoded)
        self.encoder_index = len(Model(self.input,self.encoded).layers)
#         self.encoder_index = 12
        self.prepare_models()
        
    def _encode(self, x_in):

        x,self.x_shape = cond2d_arch_encoder(x_in,
                                             2*self.latent_dim,
                                             self.n_conv,
                                             activation = self.activation,
                                             filters = self.filters,
                                             filter_factor = self.filter_factor,
                                             kernel_size = self.kernel_size,
                                             strides = self.strides_en,
                                             pool_size = self.pool_size,
                                             activity_regularizer = self.activity_regularizer)

        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        encoded = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        return encoded, z_mean, z_log_var
    
    def _decode(self, encoded):

        decoded = cond2d_arch_decoder(encoded,
                                      self.input_dim_x,
                                      self.input_dim_y,
                                      self.n_conv,
                                      self.x_shape,
                                      activation = self.activation,
                                      filters = self.x_shape[-1],
                                      filter_factor = self.filter_factor,
                                      kernel_size = self.kernel_size,
                                      strides = self.strides_de,
                                      upsample_size = self.upsample_size,
                                      activity_regularizer = self.activity_regularizer,
                                      transpose = self.transpose)
        
        return decoded

    def train(self, x, 
              training_epochs=10, 
              lr_sch=True, 
              early_stop=True,
              verbose=True): 
        if x.ndim==3:
            x = np.expand_dims(x,-1)
        super().train(x, 
                      training_epochs=training_epochs, 
                      lr_sch=lr_sch, 
                      early_stop=early_stop,
                      verbose=verbose)
        
    def encoder(self, x):
        if x.ndim==3:
            x = np.expand_dims(x,-1)
        assert (x.shape[1]==self.input_dim_x and x.shape[2]==self.input_dim_y),'Input dimension problem!'
        return self._encoder.predict(x)
















