from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def nearest_neighbor(components, X, distance='L1'):

	# Initializing the tensor flow variables
	init = tf.global_variables_initializer()

	# Launch the session
	sess = tf.InteractiveSession()
	sess.run(init)

	x_vals_train = components
	x_vals_test = X
	n_d = components.shape[1]

	# Placeholders
	x_data_train = tf.placeholder(shape=[None, n_d], dtype=tf.float32)
	x_data_test = tf.placeholder(shape=[None, n_d], dtype=tf.float32)

	if distance=='L1':
		distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)

	if distance=='L2':
		distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))

	prediction = tf.argmin(distance, axis=1)

	nn_labels = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_vals_test})

	return nn_labels

def weight(y_s):
    vp = np.piecewise(y_s,[y_s < 0, y_s > 0],[np.nan, 1])
    n = np.isnan(vp)
    c = np.cumsum(~n)

    d = np.diff(np.concatenate(([0.], c[n])))
    vp[n] = -d

    vm = np.piecewise(y_s,[y_s > 0, y_s < 0],[np.nan, 1])
    n = np.isnan(vm)
    c = np.cumsum(~n)
    d = np.diff(np.concatenate(([0.], c[n])))
    vm[n] = -d

    return np.cumsum(vp)+np.cumsum(vm)

def weight_over_axes(y_s):
    return np.apply_along_axis(weight, 2, y_s)

def distance_nn_old(components, X, metric='L1'):

	# Initializing the tensor flow variables
	init = tf.global_variables_initializer()

	# Launch the session
	sess = tf.InteractiveSession()
	sess.run(init)

	x_vals_train = components
	x_vals_test = X
	n_d = components.shape[1]

	# Placeholders
	x_data_train = tf.placeholder(shape=[None, n_d], dtype=tf.float32)
	x_data_test = tf.placeholder(shape=[None, n_d], dtype=tf.float32)

	if metric=='L1':
		distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)

	if metric=='L2':
		distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))

	if metric=='L4':
		distance = tf.sqrt(tf.sqrt(tf.reduce_sum(tf.square(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)))), reduction_indices=1)))

	if metric=='exp':
		distance = tf.sqrt(tf.reduce_sum(tf.exp(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)))), reduction_indices=1))
#		distance = tf.reduce_sum(tf.exp(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)))), axis=2)

	if metric=='weight':
		weights = tf.py_func(weight_over_axes, [tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))], tf.float32)
		distance = tf.reduce_sum(tf.multiply(weights,tf.exp(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))))), axis=2)

	dist_pred = tf.reduce_min(distance, reduction_indices=[1])

	des_out = sess.run(dist_pred, feed_dict={x_data_train: x_vals_train, x_data_test: x_vals_test})
	sess.close()

	return des_out

def distance_nn(components, X, metric='L1'):

	# Launch the session
	sess = tf.InteractiveSession()

	n_d = components.shape[1]
	num = X.shape[0]
	n_divide = int(num/999)+1
	dis_out = np.zeros(num)

	# Placeholders
	x_data_train = tf.placeholder(shape=[None, n_d], dtype=tf.float32)
	x_data_test = tf.placeholder(shape=[None, n_d], dtype=tf.float32)

	if metric=='L1':
		all_distances = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)
		min_distance = tf.reduce_min(all_distances, reduction_indices=[1])
		for inds in np.array_split(np.arange(num), n_divide):
			dis_out[inds] = sess.run(min_distance, feed_dict={x_data_train: components, x_data_test: X[inds]})

	if metric=='L2':
		all_distances = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))
		min_distance = tf.reduce_min(all_distances, reduction_indices=[1])
		for inds in np.array_split(np.arange(num), n_divide):
			dis_out[inds] = sess.run(min_distance, feed_dict={x_data_train: components, x_data_test: X[inds]})

	if metric=='L4':
		all_distances = tf.sqrt(tf.sqrt(tf.reduce_sum(tf.square(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)))), reduction_indices=1)))
		min_distance = tf.reduce_min(all_distances, reduction_indices=[1])
		for inds in np.array_split(np.arange(num), n_divide):
			dis_out[inds] = sess.run(min_distance, feed_dict={x_data_train: components, x_data_test: X[inds]})

	if metric=='exp':
		all_distances = tf.sqrt(tf.reduce_sum(tf.exp(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)))), reduction_indices=1))
#		distance = tf.reduce_sum(tf.exp(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)))), axis=2)
		min_distance = tf.reduce_min(all_distances, reduction_indices=[1])
		for inds in np.array_split(np.arange(num), n_divide):
			dis_out[inds] = sess.run(min_distance, feed_dict={x_data_train: components, x_data_test: X[inds]})

	sess.close()

	return dis_out


