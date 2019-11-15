from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as ospath
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adadelta
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def step_decay_schedule(initial_lr=1e-1, decay_factor=0.5, step_size=1, verbose=0):
    def schedule(epoch):
        lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        return lr
    return LearningRateScheduler(schedule,verbose=verbose)
    

class AutoEncoderBase(object):
    def __init__(self,
                 input_dim = None,
                 latent_dim = None,
                 activation='relu',
                 activity_regularizer=None,
                 learning_rate=0.03,
                 batch_size=100,
                 variational = False,
                 debug_mode = False):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.activity_regularizer = activity_regularizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.variational = variational
        self.epoch_total = 0
        
        if debug_mode:
            self.input = Input(shape=(self.input_dim))
            self.encoded = self._encode(self.input)
            self.decoded = self._decode(self.encoded)
            self.encoder_index = 1
            self.prepare_models()
        
    def _encode(self, x_in):
        encoded = x_in
        return encoded
    
    def _decode(self, encoded):
        decoded = encoded
        return decoded
        
    def prepare_models(self, optimizer=None, loss=None):

        self.autoencoder = Model(self.input, self.decoded)
        self._encoder = Model(self.input, self.encoded)
        self.z_input = Input(shape=(self.latent_dim,))
        self.x_output = self.z_input
        for layer in self.autoencoder.layers[self.encoder_index:]:
            self.x_output = layer(self.x_output)
        
        self._decoder = Model(self.z_input, self.x_output)
        
        if optimizer is None:
            self.optimizer = Adadelta(self.learning_rate)
        else:
            self.optimizer = optimizer
            
        if loss is None:
            self.loss = mse
        else:
            self.loss = loss
            
        if not self.variational:
            self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
        else:
            self.reconstruction_loss = self.input_dim*self.loss(K.flatten(self.input), K.flatten(self.decoded))
            self.kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
            self.kl_loss = -0.5*K.sum(self.kl_loss, axis=-1)
            self.vae_loss = K.mean(self.reconstruction_loss + self.kl_loss)
            self.autoencoder.add_loss(self.vae_loss)
            self.autoencoder.compile(optimizer=self.optimizer)

    def summary(self):
        self.autoencoder.summary()
                        
    def encoder(self, x):
        assert (x.shape[1]==self.input_dim),'Input dimension problem!'
        return self._encoder.predict(x)

    def decoder(self, z):
        assert (z.shape[1]==self.latent_dim),'Input dimension problem!'
        return self._decoder.predict(z)

    def train(self, x, 
              training_epochs=10, 
              lr_sch=True, 
              early_stop=True,
              verbose=True): 
        
        callbacks = []
        if lr_sch:
            lr_sched = step_decay_schedule(verbose=verbose)
            callbacks += [lr_sched]
        if early_stop:
            earlystop = EarlyStopping(monitor = 'loss',
                                      min_delta = 0.01,
                                      patience = 3,
                                      verbose = verbose,
                                      restore_best_weights = True)
            callbacks += [earlystop]
        
        if self.variational:
            y = None
        else:
            y = x
            
        self.autoencoder.fit(x, y,
                             epochs=training_epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             verbose=verbose,
                             callbacks=callbacks)

    def save(self, name='model.h5', path='./'):
        self.autoencoder.save_weights(ospath(path,name))

    def restore(self, name='model.h5', path='./'):
        self.autoencoder.load_weights(ospath(path,name))

    def check(self, x, l, w=None,training_epochs=3):
        
        self.train(x,training_epochs=training_epochs)
        
        ii = np.random.randint(0,x.shape[0])
        x = x[ii:ii+1]
        z = self.encoder(x)
        xd = self.decoder(z)
        xr = self.decoder(np.random.normal(0,1,z.shape))
        print(90*'-')
        print( 'Encoded and decoded shapes are {}, {}.'.format(z.shape,xd.shape) )

        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4))

        if w is None:
            ax1.plot(x.reshape(l))
            ax2.plot(xd.reshape(l))
            ax3.plot(xr.reshape(l))          
        else:
            ax1.imshow(x.reshape(l,w))
            ax2.imshow(xd.reshape(l,w))
            ax3.imshow(xr.reshape(l,w)) 
        
        ax1.set_title('original')
        ax2.set_title('reconstructed')
        ax3.set_title('noise') 
        
        plt.show()  
        
        x2 = self.autoencoder.predict(x)
        delta = np.sum((x2-x)**2)
        print( 'Differece between method and predicdt is {}.'.format(delta) )
        print(90*'-')

    def illustration(self,data):
        plot_results((self.encoder,self.decoder),
                     data,
                     batch_size=128)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        purge = list(vars(self).keys())
        for i in purge:
            exec('del self.'+i)
        gc.collect()


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name=None):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder(x_test)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    
    plt.show()
    if not model_name is None:
        os.makedirs(model_name, exist_ok=True)
        filename = os.path.join(model_name, "vae_mean.png")
        plt.savefig(filename)
        filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    if not model_name is None:
        plt.savefig(filename)
    plt.show()








