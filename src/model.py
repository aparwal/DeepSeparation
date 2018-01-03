# model.py 
# Anand Parwal

import numpy as np
# import pickle
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Reshape, TimeDistributed,concatenate,Lambda, Permute
from keras.optimizers import RMSprop
from configuration import *
from TimeFreqMasking import TimeFreqMasking
# from kapre.time_frequency import Spectrogram

# environ['TF_CPP_MIN_LOG_LEVEL']='3'

# src = np.random.random((seq_len, sr * time_len,))
# K.set_learning_phase(1) #set learning phase

# batch_shape=(batch_size, ModelConfig.SEQ_LEN,n_bins)
def create_model(rnn_nos=1,seq_in = Input(shape=(seq_len,n_bins)),dense_size=2):

	# input2=Reshape((ModelConfig.SEQ_LEN,n_bins))(seq_in)
	# x=seq_in
	if rnn_nos==1:
		x=LSTM(512,return_sequences=False,dropout=0.3,recurrent_dropout=0.1,stateful=stateful)(seq_in)
	else:
		x=LSTM(512,return_sequences=True,dropout=0.3,recurrent_dropout=0.1)(seq_in)
		for nos in range(rnn_nos-2):
			x = LSTM(512, return_sequences=True,dropout=0.3,recurrent_dropout=0.1)(x)
		x=LSTM(512,return_sequences=False,dropout=0.3,recurrent_dropout=0.1)(x)

	# y_m=Dense((n_bins)*dense_size,activation='relu')(x)#(Flatten()(x))
	# y_v=Dense((n_bins)*dense_size,activation='relu')(x)#(Flatten()(x))
	
	music_hat=Dense((n_bins),activation='relu')(x)#(Flatten()(x))
	vocal_hat=Dense((n_bins),activation='relu')(x)#(Flatten()(x))

	Music_pred=TimeFreqMasking(name='Music_pred')([music_hat,vocal_hat,seq_in])
	Vocal_pred=TimeFreqMasking(name='Vocal_pred')([vocal_hat,music_hat,seq_in])

	# Because keras.layer.TimeDistributed does not support multiple output yet!
	# merged_output=concatenate([Music_pred,Vocal_pred],axis=1,name='merged_output')

	model=Model(inputs=seq_in, outputs=[Music_pred,Vocal_pred])
	# model.compile(loss=[penalized_loss(Vocal_pred,regular_const[0]),penalized_loss(Music_pred,regular_const[1])],loss_weights=loss_weights,
	# 																	 optimizer='RMSprop', metrics=['accuracy'])
	model.compile(loss=['mse','mse'],loss_weights=loss_weights,
																		 optimizer='nadam', metrics=['accuracy'])

	# modelPerSeq.summary()

	return model

def penalized_loss(noise,regular_const=0.5):

	def loss_general(y_true, y_pred):
		return K.mean(K.square(y_pred - y_true) - regular_const*K.square(y_true - noise), axis=-1)
		
	def loss(y_true, y_pred):
		loss1=loss_general(y_true, y_pred)
		# def f1(): return K.variable(0)
		# def f2(): return loss1
		# loss = K.switch(K.less(loss1, K.variable(0)), f1, f2)
		return loss1

	return loss

# @staticmethod
# def batch_to_spec(src, num_wav):
# 	# shape = (batch_size, n_frames, n_freq) => (batch_size, n_freq, n_frames)
# 	batch_size, seq_len, freq = src.shape
# 	src = np.reshape(src, (num_wav, -1, freq))
# 	src = src.transpose(0, 2, 1)
# 	return src

if __name__ == "__main__":
	model=create_model()
	model.summary()
	# model()