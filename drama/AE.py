import numpy as np
import gc
import tensorflow as tf
from utils import *

class AutoEncoder(object):
	def __init__(self, network_architecture, activition=tf.nn.relu, 
		           learning_rate=0.001, batch_size=100):
		self.nn_en, self.nn_de = network_architecture[0], network_architecture[1]
		self.activition = activition
		self.epoch_total = 0
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.n_input = self.nn_en[0]
		self.n_z = self.nn_en[-1]

		assert self.nn_en[0]==self.nn_de[-1],"Input and output dimension should be match in encoder and decoder!"

		assert self.nn_en[-1]==self.nn_de[0],"Latent layer dimension should be match in encoder and decoder!"

		# tf Graph input
		self.X = tf.placeholder(tf.float32, [None, self.n_input])

		# tf Graph codec
#		self.Z = tf.placeholder(tf.float32, [None, self.n_z])

		self.net = self._net_initializer(self.nn_en,self.nn_de)

		self.encoder_op = self._encoder(self.X, self.net['weights_encode'],self.net['biases_encode'])

		self.Z = self.encoder_op
		self.decoder_op = self._decoder(self.Z, self.net['weights_decode'],self.net['biases_decode'])

		# Prediction
		self.y_pred = self.decoder_op
		# Targets (Labels) are the input data.
		self.y_true = self.X

		# Define loss and optimizer, minimize the squared error
		self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)

		# Initializing the tensor flow variables
		init = tf.global_variables_initializer()
	#         init = tf.initialize_all_variables()

		# Launch the session
		self.sess = tf.InteractiveSession()
		self.sess.run(init)

	def __enter__(self):
		  return self

	def __exit__(self, type, value, traceback):
		del self.nn_en
		del self.nn_de
		del self.activition
		del self.epoch_total
		del self.learning_rate
		del self.batch_size
		del self.n_input
		del self.n_z	
		del self.X
		del self.net
		del self.encoder_op
		del self.Z
		del self.decoder_op
		del self.y_pred
		del self.y_true
		del self.cost
		del self.optimizer
		self.sess.close()
		del self.sess
		gc.collect()

	def _make_weights(self, nn, prefix): 
		w = dict()
		for i in range(len(nn)-1):
			w[prefix+'_w_'+str(i+1)]=tf.Variable(tf.random_normal((nn[i],nn[i+1])))

		return w

	def _make_biases(self, nn, prefix): 
		b = dict()
		for i in range(1,len(nn)):
			b[prefix+'_b_'+str(i)]=tf.Variable(tf.zeros([nn[i]], dtype=tf.float32))   
		return b

	def _net_initializer(self, nn_en, nn_de):
		all_weights = dict()
		all_weights['weights_encode'] = self._make_weights(nn_en,'encoder')
		all_weights['biases_encode'] = self._make_biases(nn_en,'encoder')

		all_weights['weights_decode'] = self._make_weights(nn_de,'decoder')
		all_weights['biases_decode'] = self._make_biases(nn_de,'decoder')
		return all_weights           
		      
	def _encoder(self, X, weights, biases):
		i = 1
		layer = self.activition(tf.add(tf.matmul(X, weights['encoder_w_'+str(i)]), biases['encoder_b_'+str(i)]))
		n_layer = len(self.nn_en)
		for i in range(2,n_layer):
			layer = self.activition(tf.add(tf.matmul(layer, weights['encoder_w_'+str(i)]),biases['encoder_b_'+str(i)]))
		return layer

	def _decoder(self, Z, weights, biases):
		i = 1
		layer = self.activition(tf.add(tf.matmul(Z, weights['decoder_w_'+str(i)]), biases['decoder_b_'+str(i)]))
		n_layer = len(self.nn_de)
		for i in range(2,n_layer):
			layer = self.activition(tf.add(tf.matmul(layer, weights['decoder_w_'+str(i)]), biases['decoder_b_'+str(i)]))
		return layer
		      		  
	def encoder(self, X):
		return self.sess.run(self.Z, feed_dict={self.X: X})

	def decoder(self, Z):
		return self.sess.run(self.decoder_op,feed_dict={self.Z: Z})

	def train(self, X, training_epochs=10, verbose=True): 
		n_samples = X.shape[0]

		total_batch = int(n_samples / self.batch_size)
		# Training cycle
		for epoch in range(training_epochs):
			# Loop over all batches
			c = 0
			for i in range(total_batch):
				batch_xs = self.get_batch(self.batch_size,X)
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_xs})
			# Display logs per epoch step
			self.epoch_total = self.epoch_total+1

			if verbose:
				flushout('Epoch: %d, cost= %5g',(self.epoch_total, c))

		if verbose:
			print 

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


