from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gc
import tensorflow as tf
from .utils import *
from tensorflow.contrib.layers import xavier_initializer

class VariationalAutoEncoder(object):

	def __init__(self,  network_architecture, 
											transfer_fct=tf.nn.relu,
											initializer = xavier_initializer(),
		           				learning_rate = 0.001):

		self.nn_rec, self.nn_gen = network_architecture[0], network_architecture[1]
		self.transfer_fct = transfer_fct
		self.learning_rate = learning_rate
		self.initializer = initializer

		assert self.nn_rec[0]==self.nn_gen[-1],"Input and output dimension should be match in encoder and decoder!"

		assert self.nn_rec[-1]==self.nn_gen[0],"Latent layer dimension should be match in encoder and decoder!"

		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, self.nn_rec[0]])

		# tf Graph codec
		self.z_in = tf.placeholder(tf.float32, [None, self.nn_rec[-1]])

		self._create_network()

		self._create_loss_optimizer()
		
		init = tf.global_variables_initializer()

		self.sess = tf.InteractiveSession()
		self.sess.run(init)

	def _create_network(self):
		self.network_weights = self._net_initializer(self.nn_rec, self.nn_gen)

		self.z_mean, self.z_log_sigma_sq = self._encode_network()

		n_z = self.nn_rec[-1]
		n_sample = tf.shape(self.x)[0]

		eps = tf.random_normal((n_sample, n_z), 0, 1, 
		                       dtype=tf.float32)
		# z = mu + sigma*epsilon
		self.z = tf.add(self.z_mean, 
		                tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

		self.x_reconstr_mean = self._decode_network()

	def __enter__(self):
		  return self

	def __exit__(self, type, value, traceback):
		del self.nn_rec
		del self.nn_gen
		del self.transfer_fct
		del self.learning_rate
		del self.initializer
		del self.x
		del self.z_in
		del self.network_weights
		del self.z_mean
		del self.z_log_sigma_sq
		del self.z
		del self.x_reconstr_mean
		del self.cost  
		del self.optimizer
		self.sess.close()
		del self.sess
		gc.collect()

	def _create_network(self):
		self.network_weights = self._net_initializer(self.nn_rec, self.nn_gen)

		self.z_mean, self.z_log_sigma_sq = self._encode_network()

		n_z = self.nn_rec[-1]
		n_sample = tf.shape(self.x)[0]

		eps = tf.random_normal((n_sample, n_z), 0, 1, 
		                       dtype=tf.float32)
		# z = mu + sigma*epsilon
		self.z = tf.add(self.z_mean, 
		                tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

		self.x_reconstr_mean = self._decode_network()


	def _make_weights(self, nn): 
		w = dict()
		w['h1']= tf.Variable(self.initializer((nn[0],nn[1])))
		for i in range(1,len(nn)-2):
		    w['h'+str(i+1)]=tf.Variable(self.initializer((nn[i],nn[i+1])))
		w['out_mean'] = tf.Variable(self.initializer((nn[-2],nn[-1])))
		w['out_log_sigma'] = tf.Variable(self.initializer((nn[-2],nn[-1])))

		return w

	def _make_biases(self, nn): 
		b = dict()
		b['b1']=tf.Variable(tf.zeros([nn[1]], dtype=tf.float32))
		for i in range(2,len(nn)-1):
		    b['b'+str(i)]=tf.Variable(tf.zeros([nn[i]], dtype=tf.float32))   
		b['out_mean'] = tf.Variable(tf.zeros([nn[-1]], dtype=tf.float32))   
		b['out_log_sigma'] = tf.Variable(tf.zeros([nn[-1]], dtype=tf.float32))   

		return b

	def _net_initializer(self, nn_rec,nn_gen):
		all_weights = dict()
		all_weights['weights_encode'] = self._make_weights(nn_rec)
		all_weights['biases_encode'] = self._make_biases(nn_rec)

		all_weights['weights_decode'] = self._make_weights(nn_gen)
		all_weights['biases_decode'] = self._make_biases(nn_gen)
		return all_weights            
		      
	def _encode_network(self):
		weights = self.network_weights["weights_encode"]
		biases = self.network_weights["biases_encode"]

		i = 1
		layer = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h'+str(i)]), 
		                                   biases['b'+str(i)]))
		n_layer = len(self.nn_rec)
		for i in range(2,n_layer-1):
		    layer = self.transfer_fct(tf.add(tf.matmul(layer, weights['h'+str(i)]), 
		                                   biases['b'+str(i)]))

		z_mean = tf.add(tf.matmul(layer, weights['out_mean']),
		                biases['out_mean'])
		z_log_sigma_sq = \
		    tf.add(tf.matmul(layer, weights['out_log_sigma']), 
		           biases['out_log_sigma'])

		return (z_mean, z_log_sigma_sq)

	def _decode_network(self):
		weights = self.network_weights["weights_decode"]
		biases = self.network_weights["biases_decode"]

		i = 1
		layer = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h'+str(i)]), 
		                                   biases['b'+str(i)]))
		n_layer = len(self.nn_gen)
		for i in range(2,n_layer-1):
		    layer = self.transfer_fct(tf.add(tf.matmul(layer, weights['h'+str(i)]), 
		                                   biases['b'+str(i)]))
		x_reconstr_mean = \
		    tf.nn.sigmoid(tf.add(tf.matmul(layer, weights['out_mean']), 
		                         biases['out_mean']))

		return x_reconstr_mean
		      
	def _create_loss_optimizer(self):

		reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
		                + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),1)

		latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
		                                   - tf.square(self.z_mean) 
		                                   - tf.exp(self.z_log_sigma_sq), 1)
		self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

	def encoder(self, X):
		return self.sess.run(self.z_mean, feed_dict={self.x: X})

	def decoder(self, z_mu):
		return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})

	def reconstruct(self, X):
		""" Use VAE to reconstruct given data. """
		return self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X})

	def train(self, X, training_epochs=10,  
						batch_size = 100,
						verbose = True): 

		n_samples = X.shape[0]
		# Training
		for epoch in range(training_epochs):
			avg_cost = 0.
			total_batch = int(n_samples / batch_size)

			for i in range(total_batch):

				batch_xs = self.get_batch(batch_size,X)

				opt, cost = self.sess.run((self.optimizer, self.cost), 
													feed_dict={self.x: batch_xs})

				avg_cost += cost / n_samples * batch_size

			if verbose:
				flushout('Epoch: %d, cost= %5g',(epoch+1, avg_cost))

		if verbose:
			print ()

	def save(self, path):
#    path = os.path.join(path, 'model.cpkt')
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, path)
#		return save_path

	def restore(self, path):
#    path = os.path.join(path, 'model.cpkt')
		saver = tf.train.Saver()
		saver.restore(self.sess, path)

	def get_batch(self, n_train,X):		
			num, n_features = X.shape
			num = X.shape[0]
			indx = np.arange(num)
			np.random.shuffle(indx)
			indx = indx[:n_train]
			
			x_batch = X[indx]
			return x_batch

