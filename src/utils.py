import os
from configuration import *
from scipy.io import wavfile
from scipy.signal import stft,check_COLA,istft
import numpy as np
import pickle


# save decoded dataset as pickle file
def save_as_wav(wavs_dir=wavs_dir):
	dataset= {
		'vocals': [],
		'accompaniment': [],
		'bass': [],
		'drums': [],
		'other':  [],
		'mixture': []
	}

	count=0
	for folder in os.listdir(os.path.join(wavs_dir,'train')):
		count+=1
		if count % 10 == 0:
			print("\rGetting Data: {0:.2f}%  ".format(count /len(os.listdir(os.path.join(wavs_dir,'train')))  * 100), end="")
		for key in dataset.keys():
			_,data=wavfile.read(os.path.join(wavs_dir,"train",folder,str(key)+".wav"))
			dataset[key].append(data[:,0])
			dataset[key].append(data[:,1])
		# mix=(np.hstack(dataset['vocals'])+np.hstack(dataset['accompaniment']))/2
		# print(mix.mean(),np.hstack(dataset['mixture']).mean())
		# print(mix.shape,np.hstack(dataset['mixture']).shape)

	print("Saving dataset")
	pickle.dump(dataset, open(wavs_dir+"/dataset.pickle", "wb"))
	print("Dataset saved")

# read pickled wav dataset
def read_data_all(infile = wavs_dir+"/dataset_stft.pickle"):
    dataset = pickle.load(open(infile, "rb"));
    return dataset['mixture'],dataset['vocals'],dataset['accompaniment'],dataset['drums'],dataset['bass'],dataset['other']

# read pickled wav dataset
def read_data(infile = wavs_dir+"/dataset_stft.pickle"):
    dataset = pickle.load(open(infile, "rb"));
    return dataset['mixture'],dataset['vocals'],dataset['accompaniment']

def make_chunks(lis):
	arr=np.hstack(lis)
	chunk_len=len(arr)//int(sr*time_len)*int(sr*time_len)
	return arr[:chunk_len].reshape(-1,int(sr*time_len))

def make_stft(lis):
	arr=make_chunks(lis)
	mags=[]
	angles=[]
	if check_COLA('hann',nperseg=perseg,noverlap = overlap):
		for wav in arr:
			f,t,X=stft(wav,nperseg=perseg,noverlap = overlap)
			mags.append(np.transpose(np.abs(X)))
			angles.append(np.transpose(np.angle(X)))
	else:
		print("COLA constraint not met, in func: utils.make_stft")
		exit()

	# print(len(mags),np.abs(mags[0].shape))
	return np.vstack(mags),np.vstack(angles)

def get_stft_matrix(magnitudes, phases):
	return magnitudes * np.exp(1.j * phases)

def to_wav(mag, phase, overlap=overlap):
	mag=np.transpose(mag)
	phase=np.transpose(phase)
	stft_maxrix = get_stft_matrix(mag, phase)
	# print(stft_maxrix.shape)
	a=[]
	for mat in stft_maxrix:
		# print(mat.shape)
		a.append(istft(mat,fs=sr, noverlap=overlap))
		# print("one ",end="")
	# print(np.array(a).shape)
	return np.array(a)




def save_as_stft(wavs_dir = wavs_dir):
	mix,voc,acc,dru,bas,oth=read_data_all(infile = wavs_dir+"/dataset.pickle")
	dataset_stft={}
	dataset_stft['mixture'],dataset_stft['mixturea']=make_stft(mix)
	dataset_stft['vocals'],dataset_stft['vocalsa']=make_stft(voc)
	dataset_stft['accompaniment'],dataset_stft['accompanimenta']=make_stft(acc)
	dataset_stft['bass'],dataset_stft['bassa']=make_stft(dru)
	dataset_stft['drums'],dataset_stft['drumsa']=make_stft(bas)
	dataset_stft['other'],dataset_stft['othera']=make_stft(oth)

	print("Saving dataset")
	pickle.dump(dataset_stft, open(wavs_dir+"/dataset_stft.pickle", "wb"))
	print("Dataset saved")

if __name__ == '__main__':
	# save_as_pickle(wavs_dir)
	save_as_stft(wavs_dir)

