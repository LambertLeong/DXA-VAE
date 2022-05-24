### basic ###
import os, sys, shutil, math, time,ast
import random as rnd
import numpy as np

### deep learning ###
import tensorflow as tf
from tensorflow.keras import layers,losses
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

### Custom ###
import utils, DXA_utils

class VAE(object):
	def __init__(self, config):
		self.config = config
		height, width, num_channels = 150, 109, 6
		last_height=int(config['vae']['last_height'])
		last_width=int(config['vae']['last_width'])
		last_channel=int(config['vae']['last_channel'])
		latent_dim=last_height*last_width*last_channel
		reshape_dim=(last_height,last_width,last_channel)
		self.im_input = layers.Input(shape=(height,width,num_channels))
		model_path=config['vae']['model_path']
		if 'None' in model_path: 
			encoder=self.build_encoder(self.im_input,latent_dim)#TODO,self.config['vae']['weights'])
			self.Model=self.attach_generator(self.im_input,encoder,reshape_dim,int(config['vae']['upscale']))
		else:
			from tensorflow import keras
			self.Model=keras.models.load_model(model_path,custom_objects={'tf': tf})

	def build_encoder(self,im_input,latent_dim,transfer_weigths=None):
		densenet = tf.keras.applications.DenseNet121(include_top=False, weights=transfer_weigths)
		x = layers.Conv2D(3, (1, 1), padding="same")(im_input)
		densenet = densenet(x)
		flat = layers.Flatten()(densenet)#(e_last)
		mu = layers.Dense(latent_dim, name='mu')(flat)
		sigma = layers.Dense(latent_dim, name="sigma")(flat)
		latent_space = layers.Lambda(utils.compute_latent, output_shape=(latent_dim,), name='latent')([mu, sigma])
		#encoder = tf.keras.models.Model(im_input, latent_space,name="dxa_encoder")
		return latent_space

	def attach_generator(self,im_input,latent_space,reshape_dim,upscale):

		x = layers.Reshape(reshape_dim)(latent_space)
		x = layers.PReLU(shared_axes = [1,2])(x)
		filters=reshape_dim[2]
		print('filters',filters)
		for i in range(5):
			x = layers.Conv2D(filters, (1,1), padding="same")(x)
			if i !=0:
				x = layers.Conv2D(filters, (1,2), padding="same")(x)
			x = layers.PReLU(shared_axes = [1,2])(x)
			#x=tf_resize(x.shape,upscale)(x)
			x=utils.tf_resize(x.shape,upscale)(x)
			x = layers.Conv2D(filters, (2,2), padding="same")(x)
			x = layers.PReLU(shared_axes = [1,2])(x)
			x = layers.Conv2D(filters, (4,4), padding="same")(x)
			x = layers.PReLU(shared_axes = [1,2])(x)
			if i>2:
				x = layers.Conv2D(filters, (8,8), padding="same")(x)
				x = layers.PReLU(shared_axes = [1,2])(x)
			filters //= upscale

		x = layers.Conv2D(filters, (8,8), padding="same")(x)
		x = layers.Cropping2D(cropping=((5, 5), (9, 10)))(x)
		x = layers.Conv2D(6, (1, 1), padding="same")(x)
		#gen = tf.keras.models.Model(gen_input, x, name="dxa_gen")
		vae = tf.keras.models.Model(self.im_input, x, name="dxa_vae",)
		return vae

class Discriminator(object):
	def __init__(self,config):
		self.config = config
		filters=int(config['disc_loss']['disc_filters'])
		alpha=float(config['disc_loss']['disc_alpha'])
		im_input = layers.Input(shape=(150,109,6)) #Hardcoded for now
		x = layers.Conv2D(filters, (3,3), padding="same")(im_input)
		x = layers.LeakyReLU(alpha)(x)
		for i in range(5):
			x = layers.Conv2D(filters, (3,3), padding="same")(x)
			x = layers.LeakyReLU(alpha)(x)
			x = layers.Conv2D(filters, (3,3),strides=2, padding="same")(x)
			x = layers.BatchNormalization(momentum=0.5)(x)
			filters*=2
		x = layers.Flatten()(x)
		x = layers.Dense(filters)(x)
		x = layers.LeakyReLU(alpha)(x)
		validity = layers.Dense(1, activation='sigmoid')(x)
		self.Model = tf.keras.models.Model(im_input, validity,name="disc")

class Perceptual_loss(object):
	def __init__(self,config):
		self.config = config
		self.selected_layers=ast.literal_eval(config['perceptual_loss']['selected_layers'])
		self.selected_layer_weights=ast.literal_eval(config['perceptual_loss']['selected_layer_weights'])
		self.transform_weight=float(config['perceptual_loss']['transform_weight'])
		self.recon_weight=float(config['perceptual_loss']['recon_weight'])
		lossModel = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
		lossModel.trainable=False
		for layer in lossModel.layers:
			layer.trainable=False
		selectedOutputs = [lossModel.layers[i].output for i in self.selected_layers]
		self.Model = tf.keras.models.Model(lossModel.inputs,selectedOutputs)

	#@staticmethod
	def perceptual_loss(self, input_image , reconstruct_image):
		lossModel=self.Model
		rc_loss = 0.0
		transform_loss = 0.0
		#transform_loss2= 0.0
		if self.config['perceptual_loss'].getboolean('dxa_loss'):
			act_ims=DXA_utils.process_hologic(input_image)
			rec_ims=DXA_utils.process_hologic(reconstruct_image)
			for i in range(len(act_ims)):
				h1_list = lossModel(act_ims[i])
				h2_list = lossModel(rec_ims[i]) 
				for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights[1:]):
					h1 = K.batch_flatten(h1)
					h2 = K.batch_flatten(h2)
					transform_loss = transform_loss + (weight * K.sum(K.abs(h1 - h2), axis=-1))
		### 6 phase loss ###
		h1_list = lossModel(input_image[:,:,:,:3])
		h2_list = lossModel(reconstruct_image[:,:,:,:3])
		#rc_loss = 0.0
		for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights):
			h1 = K.batch_flatten(h1)
			h2 = K.batch_flatten(h2)
			#rc_loss = rc_loss + weight * K.sum(K.square(h1 - h2), axis=-1) #l2
			rc_loss = rc_loss + weight * K.sum(K.abs(h1 - h2), axis=-1) #l1
			
		h1_list = lossModel(input_image[:,:,:,-3:])
		h2_list = lossModel(reconstruct_image[:,:,:,-3:])
		for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights):
			h1 = K.batch_flatten(h1)
			h2 = K.batch_flatten(h2)
			#rc_loss = rc_loss + weight * K.sum(K.square(h1 - h2), axis=-1) #l2
			rc_loss = rc_loss + weight * K.sum(K.abs(h1 - h2), axis=-1) #l1
		#'''
		return self.recon_weight*(rc_loss)+(transform_loss)*self.transform_weight#TODO:add loss weights#+transform_loss2
