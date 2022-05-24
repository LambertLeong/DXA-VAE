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
import matplotlib.pyplot as plt

### custom ###
import utils, DXA_utils, models

class train_VAE(object):
	def __init__(self,config):
		self.config=config
		self.lr=float(config['train']['learning_rate'])
		self.disc_lr=float(config['train']['disc_lr'])
		self.beta_1=self.lr*float(config['train']['beta'])
		self.disc_weight, self.pl_weight=float(config['train']['disc_weight']),float(config['train']['pl_weight'])
		self.batch_size,self.epochs = 2**int(config['train']['batch_size_power']),int(config['train']['epochs'])
		### get models ###
		self.vae=models.VAE(config)
		self.disc=models.Discriminator(config)
		self.perceptual_loss=models.Perceptual_loss(config)
		### combine models ###
		self.gan=self.create_gan(self.vae.Model, self.disc.Model, self.vae.im_input)
		### compile Models ###
		self.gan.compile(loss={'disc':'binary_crossentropy','dxa_vae':self.perceptual_loss.perceptual_loss}, 
                  loss_weights= [self.disc_weight, self.pl_weight], 
                  optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.beta_1, amsgrad=True,))
		self.disc.Model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=self.disc_lr, amsgrad=True,)
		             ,metrics=['accuracy'])
		self.total_loss=[]
		self.total_gloss = []
		self.val_loss=[]
	def create_gan(self,gen_model, disc_model,org_dxa):
		gen_img = gen_model(org_dxa)
		disc_model.trainable = False
		validity = disc_model(gen_img)
		return tf.keras.models.Model([org_dxa],[validity,gen_img])

	def train(self,dxa_train, dxa_val):

		batch_size=self.batch_size
		### data generator ###
		self.datagen = ImageDataGenerator(
			rotation_range=int(self.config['aug']['rotation_range']),
			width_shift_range=float(self.config['aug']['width_shift_range']),
			height_shift_range=(float(self.config['aug']['height_shift_range_up']),float(self.config['aug']['height_shift_range_down'])),
			horizontal_flip=self.config['aug'].getboolean('horizontal_flip'),
			vertical_flip=self.config['aug'].getboolean('vertical_flip'),
			zoom_range=(0.05),
			fill_mode="nearest",
			preprocessing_function = DXA_utils.random_blackout
			)
		data = self.datagen.flow(dxa_train,y=dxa_train, batch_size=batch_size)
		gen_label_og = np.zeros((batch_size, 1))
		real_label_og = np.ones((batch_size,1))
		real_label_val = np.ones((dxa_val.shape[0],1))
		val_img=dxa_val[7:8]
		for e in range(self.epochs):
			g_losses = []
			d_losses = []
			print('epoch:',e+1)
			for b in range(int(dxa_train.shape[0]/batch_size)):
				aug_dxa, org_dxa = next(data)
				num_samp = aug_dxa.shape[0]
				gen_label=gen_label_og[:num_samp]
				real_label=real_label_og[:num_samp]
				gen_imgs = self.vae.Model.predict_on_batch(aug_dxa)
				#Dont forget to make the discriminator trainable
				self.disc.Model.trainable = True
				#Train the discriminator
				d_loss_gen = self.disc.Model.train_on_batch(gen_imgs, gen_label)
				d_loss_real = self.disc.Model.train_on_batch(org_dxa,real_label)
				self.disc.trainable = False
				d_loss = np.add(d_loss_gen, d_loss_real)
				#Train the generator
				g_loss, _, _ = self.gan.train_on_batch([aug_dxa],[real_label, org_dxa])
				d_losses.append(d_loss)
				g_losses.append(g_loss)
			### loss###
			g_losses = np.array(g_losses)
			d_losses = np.array(d_losses)
			g_loss = np.sum(g_losses, axis=0) / len(g_losses)
			d_loss = np.sum(d_losses, axis=0) / len(d_losses)
			self.total_loss+=[[g_loss,d_loss[0],d_loss[1]]]
			self.total_gloss+=[g_loss]
			### val loss###
			valres=self.gan.evaluate(dxa_val,[real_label_val,dxa_val])
			self.val_loss+=[valres]

			if int(self.config['verbose']['show_plot']):
				fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12,6))
				fig.tight_layout()
				# plot progress
				p = self.vae.Model.predict(val_img)
				if e> 0 and e%100==0:
						clear_output(wait=True)
				for i, axes in enumerate(ax):
					for j,z in enumerate(axes):
						if i == 0:
							if j==0:
								image = p[0][:,:,j+1]/p[0][:,:,j]
							else:
								image = p[0][:,:,j+1]
						elif i == 1:
							if j==0:
								image = val_img[0][:,:,j+1]/val_img[0][:,:,j]               
							else:
								image = val_img[0][:,:,j+1]
						elif i == 2:
							image = np.abs(val_img[0][:,:,j] - p[0][:,:,j])
						if i == 2:
							ax[i,j].imshow(image, vmin=0, vmax=.05)
						else:
							ax[i,j].imshow(image)
						ax[i,j].axis('off')
				fig.subplots_adjust(wspace=0.5)
				fig.subplots_adjust(hspace=.5)
				plt.show()
