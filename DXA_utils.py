import random as rnd
import numpy as np
from tensorflow.keras import backend as K


### LOAD NPYS ###
def load_npys(list_of_paths): ### system has 1TB or RAM. Use flow from methods on smaller systems
    out = []
    pad=np.zeros((150,1,6))
    for i in list_of_paths:
        npy=np.load(i)/16383
        npy= npy[:,:,:]
        out+=[npy]
    out = np.stack(out)
    return out

### AUGMENTATION FUNC FOR DXA ###
def random_blackout(image,min_channels=4,boxmin=5,boxmax=75):
    black_height, black_width = rnd.randint(boxmin,boxmax),rnd.randint(boxmin,boxmax)
    blackbox_one = np.ones((black_height, black_width))
    im_height, im_width, num_channels = image.shape
    height, width = im_height, im_width
    channel_list = rnd.sample(range(num_channels),rnd.randint(min_channels,num_channels))
    for ch in channel_list:
        black_height, black_width = rnd.randint(boxmin,boxmax),rnd.randint(boxmin,boxmax)
        blackbox_one = np.ones((black_height, black_width))
        height_bound, width_bound = im_height-black_height, im_width-black_width
        while height > (height_bound) and width > (width_bound):
            height, width = rnd.randint(0,height_bound), rnd.randint(0,width_bound)
        flat=image[height:height+black_height,width:width+black_width,ch].flatten()
        np.random.shuffle(flat)
        image[height:height+black_height,width:width+black_width,ch]=flat.reshape((black_height, black_width))
        height,width = im_height,im_width
    return image 

### GEN DXA R Files ###
def process_hologic(input_image):
	min_clip=1/(2**14-1) #TODO: hardcoded for now
	### calculations unrolled for now, list and arrays can clean this and make it shorter ###
	atten_air_hi, atten_air_lo = input_image[:,:,:,0],input_image[:,:,:,1]
	atten_bone_hi, atten_bone_lo = input_image[:,:,:,2],input_image[:,:,:,3]
	atten_tissue_hi, atten_tissue_lo = input_image[:,:,:,4],input_image[:,:,:,5]
	### Supposed to be background corrected phases, we assume it is negligible for now ###
	calib_tissue_hi = atten_tissue_hi - atten_air_hi
	calib_tissue_lo = atten_tissue_lo - atten_air_lo
	calib_bone_hi = atten_bone_hi - atten_air_hi
	calib_bone_lo = atten_bone_lo - atten_air_lo
	### K factor search ignored because it is not differenctialbe ###
	atten_bone_air_hi = atten_air_hi + calib_bone_hi
	atten_bone_tissue_hi = atten_tissue_hi + calib_bone_hi
	atten_bone_air_lo = atten_air_lo + calib_bone_lo
	atten_bone_tissue_lo = atten_tissue_lo + calib_bone_lo
	atten_tissue_air_hi = atten_air_hi + calib_tissue_hi
	atten_tissue_bone_hi = atten_bone_hi + calib_tissue_hi
	atten_tissue_air_lo = atten_air_lo + calib_tissue_lo
	atten_tissue_bone_lo = atten_bone_lo + calib_tissue_lo
	### transphase images assumed to be 1 and K image division not performed ###
	atten_bone_hi = atten_bone_hi + calib_bone_hi;
	atten_bone_lo = atten_bone_lo + calib_bone_lo;
	atten_tissue_hi = atten_tissue_hi + calib_tissue_hi;
	atten_tissue_lo = atten_tissue_lo + calib_tissue_lo;
	### avoid divide by zero erros ###
	atten_air_hi = K.clip(atten_air_hi, min_clip, 1.0)
	atten_air_lo = K.clip(atten_air_lo, min_clip, 1.0)
	atten_air_bone_hi = K.clip(atten_bone_hi, min_clip, 1.0) 
	atten_air_bone_lo = K.clip(atten_bone_lo, min_clip, 1.0)
	atten_air_tissue_hi = K.clip(atten_tissue_hi, min_clip, 1.0)
	atten_air_tissue_lo = K.clip(atten_tissue_lo, min_clip, 1.0)
	atten_tissue_hi = K.clip(atten_tissue_hi, min_clip, 1.0)
	atten_tissue_lo = K.clip(atten_tissue_lo, min_clip, 1.0)
	atten_tissue_air_hi = K.clip(atten_tissue_air_hi, min_clip, 1.0)
	atten_tissue_air_lo = K.clip(atten_tissue_air_lo, min_clip, 1.0)
	atten_tissue_bone_hi = K.clip(atten_tissue_bone_hi, min_clip, 1.0)
	atten_tissue_bone_lo = K.clip(atten_tissue_bone_lo, min_clip, 1.0)
	atten_bone_hi = K.clip(atten_bone_hi, min_clip, 1.0)
	atten_bone_lo = K.clip(atten_bone_lo, min_clip, 1.0)
	atten_bone_air_hi = K.clip(atten_bone_air_hi, min_clip, 1.0)
	atten_bone_air_lo = K.clip(atten_bone_air_lo, min_clip, 1.0)
	atten_bone_tissue_hi = K.clip(atten_bone_tissue_hi, min_clip, 1.0)
	atten_bone_tissue_lo = K.clip(atten_bone_tissue_lo, min_clip, 1.0)
	### calculate R-Value from low & high attenuations for the 3 "adjusted" phases
	R_air = atten_air_lo/atten_air_hi
	R_air_tissue = atten_air_tissue_lo/atten_air_tissue_hi
	R_air_bone = atten_air_bone_lo/atten_air_bone_hi
	R_bone = atten_bone_lo/atten_bone_hi
	R_bone_air = atten_bone_air_lo/atten_bone_air_hi
	R_bone_tissue = atten_bone_tissue_lo/atten_bone_tissue_hi
	R_tissue = atten_tissue_lo/atten_tissue_hi
	R_tissue_air = atten_tissue_air_lo/atten_tissue_air_hi
	R_tissue_bone = atten_tissue_bone_lo/atten_tissue_bone_hi
	### Image columns are usually alternate but we stack for efficiency ###
	air_r=K.stack([R_air,R_air_bone,R_air_tissue],axis=3)
	air_hi=K.stack([atten_air_hi,atten_bone_air_hi,atten_air_tissue_hi],axis=3)
	air_lo=K.stack([atten_air_lo,atten_bone_air_lo,atten_air_tissue_lo],axis=3)
	tissue_r=K.stack([R_tissue_air,R_tissue_bone,R_tissue],axis=3)
	tissue_r=K.stack([R_tissue_air,R_tissue_bone,R_tissue],axis=3)
	tissue_hi=K.stack([atten_tissue_air_hi,atten_tissue_bone_hi,atten_tissue_hi],axis=3)
	tissue_lo=K.stack([atten_tissue_air_lo,atten_tissue_bone_lo,atten_tissue_lo],axis=3)
	bone_r=K.stack([R_bone_air,R_bone,R_bone_tissue],axis=3)
	bone_r=K.stack([R_bone_air,R_bone,R_bone_tissue],axis=3)
	bone_hi=K.stack([atten_bone_air_hi,atten_bone_hi,atten_bone_tissue_hi],axis=3)
	bone_lo=K.stack([atten_bone_air_lo,atten_bone_lo,atten_bone_tissue_lo],axis=3)

	return air_r, air_hi, air_lo, tissue_r, tissue_hi, tissue_lo, bone_r, bone_hi, bone_lo
