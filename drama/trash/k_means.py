from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class K_Means(object):

	def __init__(self, sample_values, n_clusters = 2):
		self.sample_values = sample_values
		self.n_clusters = n_clusters
		self.sample = tf.Variable(tf.zeros(sample_values.shape, dtype=tf.float32),name="sample")        
		self.n_samples = tf.shape(self.sample)[0]
		self.centroids = tf.Variable(tf.zeros([n_clusters, 2], dtype=tf.float32),name="centroids") 


		# Initializing the tensor flow variables
		init = tf.global_variables_initializer()
		#         init = tf.initialize_all_variables()

		# Launch the session
		self.sess = tf.InteractiveSession()
		self.sess.run(init)

	def fit(self, n_try):

		random_indices = tf.random_shuffle(tf.range(0, self.n_samples))
		begin = [0,]
		size = [self.n_clusters,]
		size[0] = self.n_clusters
		centroid_indices = tf.slice(random_indices, begin, size)
		initial_centroids = tf.gather(self.sample, centroid_indices)

		initial_centroid_value = self.sess.run(initial_centroids,feed_dict={self.sample:self.sample_values})

		expanded_vectors = tf.expand_dims(self.sample, 0)
		expanded_centroids = tf.expand_dims(self.centroids, 1)
		distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
		nearest_indices = tf.argmin(distances, 0)
		nearest_indices = tf.to_int32(nearest_indices)
		partitions = tf.dynamic_partition(self.sample, nearest_indices, self.n_clusters)
		updated_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)

		buff = self.sess.run(updated_centroids,feed_dict={self.sample:self.sample_values, self.centroids:initial_centroid_value})
		for i in range(n_try):
			buff = self.sess.run(updated_centroids,feed_dict={self.sample:self.sample_values, self.centroids:buff})

		self.centroids_fit = buff
		return self.centroids_fit

	def evaluate(self, sample_values):
		# Finds the nearest centroid for each sample
		expanded_vectors = tf.expand_dims(self.sample, 0)
		expanded_centroids = tf.expand_dims(self.centroids, 1)
		distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
		nearest_indices = tf.argmin(distances, 0)

		return self.sess.run(nearest_indices,feed_dict={self.sample:sample_values, self.centroids:self.centroids_fit})

def plot_clusters(ax, all_samples, all_labels, centroids, title=None):
	#Plot out the different clusters
	#Choose a different colour for each cluster
	colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
	for i, centroid in enumerate(centroids):
		#Also plot centroid
		ax.plot(centroid[0], centroid[1], markersize=15, marker="x", color='k', mew=5)
		ax.plot(centroid[0], centroid[1], markersize=13, marker="x", color='m', mew=3)
        
    #Grab just the samples fpr the given cluster and plot them out with a new colour
	ax.scatter(all_samples[:,0], all_samples[:,1], c=all_labels, marker='.', edgecolor = 'none')
	if title is not None:
		ax.set_title(r'$'+title+'$', x=0.15, y=0.01,fontsize=10)

