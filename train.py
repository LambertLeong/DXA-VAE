import os, sys,configparser,importlib, ast,argparse
from argparse import RawTextHelpFormatter
import pandas as pd
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="4"#TODO: make sure you check and set this
import models,Trainer,DXA_utils 

def parse_arguments():
	parser = argparse.ArgumentParser(epilog="""
	list index (--config, -c)
			- path to config file 
	 """,
	formatter_class=RawTextHelpFormatter)
	parser.add_argument('--config', '-c', action='store',
	required=True, dest='config', metavar='<config file>',
	help='"path to config file"')
	#parser.add_argument('--gpu', '-g', action='store',
	#required=True, dest='gpuid', metavar='<id number for gpu>',
	#help='"id for gpu to use"')
	return parser.parse_args()

def main():
	argv = parse_arguments()
	#os.environ["CUDA_VISIBLE_DEVICES"]=argv.gpuid
	configfile = str(argv.config)
	config = configparser.ConfigParser()
	config.read(configfile)	

	### Check for save path ###
	save_path=config['vae']['save_path']
	if not'.h5' in save_path:
		print('Set a proper save path. Dont Waste Resources')
		sys.exit(-1)

	### Load Data ###
	data_dir=config['data']['path']
	train_tab,val_tab,test_tab=pd.read_csv(data_dir+'train.csv'),pd.read_csv(data_dir+'val.csv'),pd.read_csv(data_dir+'test.csv')
	dxa_train=DXA_utils.load_npys(train_tab['paths'].iloc[:100])
	dxa_val=DXA_utils.load_npys(val_tab['paths'].iloc[:100])
	dxa_test=DXA_utils.load_npys(test_tab['paths'].iloc[:100])


	vae_trainer=Trainer.train_VAE(config)
	vae_trainer.train(dxa_train,dxa_val)
	vae_trainer.vae.Model.save(save_path)
	
if __name__ == '__main__':
	main()
