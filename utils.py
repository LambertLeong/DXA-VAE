import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

def compute_latent(x):
	mu, sigma = x
	batch = K.shape(mu)[0]
	dim = K.int_shape(mu)[1]
	eps = K.random_normal(shape=(batch,dim))
	return mu + K.exp(sigma/2)*eps

def tf_resize(in_shape,scale=2):
            new_row, new_col=in_shape[1]*scale,in_shape[2]*scale
            return layers.Lambda(lambda x: tf.image.resize_bilinear(x, (new_row,new_col), align_corners=True))

### RANDOM BLACKOUT ###
