#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH -o monaural-zero.out
#SBATCH -t 1-12:20:00
#SBATCH --gres=gpu:6
#SBATCH --mem=126000


# written by Anand Parwal
# based on Joint Optimization of Masks and Deep Recurrent Neural Networks for Source Separation by PS Huang et al

# import numpy as np
# import pickle
import sys
import os
#for setting dir on slurm
sys.path.append(os.getcwd())
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping,Callback,ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from utils import read_data,read_mix_voc_acc
import model
# from extras import to_wav, write_wav, soft_time_freq_mask, spec_to_batch, bss_eval_global
from TimeFreqMasking import TimeFreqMasking
# from kapre.time_frequency import Spectrogram
from configuration import *
from time import time
 
# from librosa.effects import split

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
# reset states after every song/batch
class ResetStatesCallback(Callback):
    def on_batch_begin(self, batch, logs={}):
    	if batch%TrainConfig.N_FRAMES//ModelConfig.SEQ_LEN==0:
        	self.model.reset_states()

'''
def deprocess(mag,phase):
	mag=mag.reshape(-1,ModelConfig.N_BINS)
	phase=phase[:len(mag)]
	mag=mag.reshape(-1,TrainConfig.N_FRAMES,ModelConfig.N_BINS).swapaxes(2,1)
	phase=phase.reshape(-1,TrainConfig.N_FRAMES,ModelConfig.N_BINS).swapaxes(2,1)
	# print("making wavs ")
	wavs=to_wav(mag,phase)
	return wavs
'''
def train(f='tempmodel.h5',limit=[0,10]):

	mix,voc,acc=read_mix_voc_acc(limit=limit)
	

	newmodel=model.create_model()
	newmodel.summary()
	# newmodel.compile(loss=['mse','mse'],loss_weights=ModelConfig.loss_weights,
	# 	optimizer='RMSprop', metrics=['accuracy'])

	# states_resetter = ResetStatesCallback()
	tensorbd  = TensorBoard(log_dir="logs/manual/{}-state-discrim_{}_{}".format(time(),regular_const[0],regular_const[1]),
		batch_size=TrainConfig.batch_size,histogram_freq=10)
	earlystop = EarlyStopping(monitor='loss',min_delta=0.001,patience=15)
	reduce_lr = ReduceLROnPlateau(monitor='loss')
	callbacks_list=[tensorbd, earlystop]#, states_resetter]

	newmodel.fit(mix.reshape(-1,seq_len,n_bins),[acc.reshape(-1,seq_len,n_bins),voc.reshape(-1,seq_len,n_bins)], 
		epochs=TrainConfig.epochs	, batch_size=TrainConfig.batch_size, 
		verbose=2, validation_split=TrainConfig.validation_split,callbacks=callbacks_list,shuffle=False)
	newmodel.save(f)

if __name__ == '__main__':
	train(limit=[0,100])
	abhishek messed here