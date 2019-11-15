from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pylab as plt
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K

from .nn_utils import *

def dense_arch(x_in,arch,activation,last_layer_activation=None,activity_regularizer=None):
    
    assert x_in.get_shape().as_list()[1]==arch[0],'The size of the input layer is not compatible with the first layer!'
    
    assert len(arch)>1,'Number of layers has to be more than 1!'
    
    if len(arch)==2:
        if last_layer_activation is None:
            x = Dense(arch[1],
                      activation=activation,
                      activity_regularizer=activity_regularizer)(x_in)
            return x
        else:
            x = Dense(arch[1],
                      activation=last_layer_activation,
                      activity_regularizer=activity_regularizer)(x_in)
            return x
    
    x = Dense(arch[1],
              activation=activation,
              activity_regularizer=activity_regularizer)(x_in)
    for i in arch[2:-1]:
        x = Dense(i,
                  activation=activation,
                  activity_regularizer=activity_regularizer)(x)
    if last_layer_activation is None:
        x = Dense(arch[-1],
                  activation=activation,
                  activity_regularizer=activity_regularizer)(x)
    else:
        x = Dense(arch[-1],
                  activation=last_layer_activation,
                  activity_regularizer=activity_regularizer)(x)
    return x


class DenseAutoEncoder(AutoEncoderBase):
    def __init__(self,
                 input_dim = None,
                 latent_dim = None,
                 network_architecture = None, 
                 activation = 'relu',
                 last_layer_activation = 'sigmoid',
                 activity_regularizer = None,
                 learning_rate = 0.03,
                 batch_size = 100):
        
        super().__init__(input_dim = input_dim,
                         latent_dim = latent_dim,
                         activation = activation,
                         activity_regularizer = activity_regularizer,
                         learning_rate = learning_rate,
                         batch_size = batch_size)
        
        if network_architecture is None:
            self.network_architecture = [ [self.input_dim,self.input_dim//2,self.latent_dim],
                                          [self.latent_dim,self.input_dim//2,self.input_dim] ]
        else:
            self.network_architecture = network_architecture
        
        self.encoder_arch = self.network_architecture[0]
        self.decoder_arch = self.network_architecture[1]
        self.input_dim = self.encoder_arch[0]
        self.latent_dim = self.encoder_arch[-1] 
        self.last_layer_activation = last_layer_activation
        
        assert len(self.encoder_arch)>1,'Encoder layers have to be more than 1!'
        assert len(self.decoder_arch)>1,'Decoder layers have to be more than 1!'
        assert self.encoder_arch[0]==self.decoder_arch[-1],\
                "Input and output dimension have to be equal in encoder and decoder!"
        assert self.encoder_arch[-1]==self.decoder_arch[0],\
                "Latent layer dimension have to be equal in encoder and decoder!"

        self.input = Input(shape=(self.input_dim))
        self.encoded = self._encode(self.input)
        self.decoded = self._decode(self.encoded)
        
        self.encoder_index = len(self.encoder_arch)
        
        self.prepare_models()
        
    def _encode(self, x_in):

        encoded = dense_arch(x_in,
                             arch = self.encoder_arch,
                             activation = self.activation,
                             activity_regularizer = self.activity_regularizer)

        return encoded
    
    def _decode(self, encoded):
        
        decoded = dense_arch(encoded,
                             arch = self.decoder_arch,
                             activation = self.activation,
                             last_layer_activation = self.last_layer_activation,
                             activity_regularizer = self.activity_regularizer)
        
        return decoded

class DenseVariationalAutoEncoder(AutoEncoderBase):
    def __init__(self,
                 input_dim = None,
                 latent_dim = None,
                 network_architecture = None, 
                 activation = 'relu',
                 last_layer_activation = 'sigmoid',
                 activity_regularizer = None,
                 learning_rate = 0.03,
                 batch_size = 100):
        
        super().__init__(input_dim = input_dim,
                         latent_dim = latent_dim,
                         activation = activation,
                         activity_regularizer = activity_regularizer,
                         learning_rate = learning_rate,
                         batch_size = batch_size,
                         variational = True)
        
        if network_architecture is None:
            self.network_architecture = [ [self.input_dim,self.input_dim//2,self.latent_dim],
                                          [self.latent_dim,self.input_dim//2,self.input_dim] ]
        else:
            self.network_architecture = network_architecture
        
        self.encoder_arch = self.network_architecture[0]
        self.decoder_arch = self.network_architecture[1]
        self.input_dim = self.encoder_arch[0]
        self.latent_dim = self.encoder_arch[-1] 
        self.last_layer_activation = last_layer_activation
        
        assert len(self.encoder_arch)>1,'Encoder layers have to be more than 1!'
        assert len(self.decoder_arch)>1,'Decoder layers have to be more than 1!'
        assert self.encoder_arch[0]==self.decoder_arch[-1],\
                "Input and output dimension have to be equal in encoder and decoder!"
        assert self.encoder_arch[-1]==self.decoder_arch[0],\
                "Latent layer dimension have to be equal in encoder and decoder!"

        self.input = Input(shape=(self.input_dim))
        self.encoded,self.z_mean, self.z_log_var = self._encode(self.input)
        self.decoded = self._decode(self.encoded)
        
#         self.encoder_index = len(self.encoder_arch)+2
        self.encoder_index = len(Model(self.input,self.encoded).layers)
        
        self.prepare_models()

    def _encode(self, x_in):

        x = dense_arch(x_in,
                       arch = self.encoder_arch[:-1],
                       activation = self.activation,
                       activity_regularizer = self.activity_regularizer)
        
        
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        encoded = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        return encoded, z_mean, z_log_var
    
    def _decode(self, encoded):

        decoded = dense_arch(encoded,
                             arch = self.decoder_arch,
                             activation = self.activation,
                             last_layer_activation = self.last_layer_activation,
                             activity_regularizer = self.activity_regularizer)

        return decoded
                
















