import os
from configuration import *
from scipy.io import wavfile
from scipy.signal import stft,check_COLA,istft
import numpy as np
import pickle
import multiprocessing as mp

# save decoded dataset as pickle file
def save_as_wav(dir_list):
	dataset= {
		'vocals': [],
		'accompaniment': [],
		'bass': [],
		'drums': [],
		'other':  [],
		'mixture': []
	}

	count=0
	for folder in dir_list:
		# if count>=10:
		# 	return dataset
		# count+=1
		# if count % 5 == 0:
		# 	print("\rGetting Data: {0:.2f}%  ".format(count /len(os.listdir(os.path.join(wavs_dir,'train')))  * 100), end="")
		for key in dataset.keys():
			_,data=wavfile.read(os.path.join(wavs_dir,"train",folder,str(key)+".wav"))
			dataset[key].append(data[:,0])
			dataset[key].append(data[:,1])
		# mix=(np.hstack(dataset['vocals'])+np.hstack(dataset['accompaniment']))/2
		# print(mix.mean(),np.hstack(dataset['mixture']).mean())
		# print(mix.shape,np.hstack(dataset['mixture']).shape)
	# print("Complete")
	return dataset
	# print("Saving dataset")
	# pickle.dump(dataset, open(wavs_dir+"/dataset.pickle", "wb"),pickle.HIGHEST_PROTOCOL)
	# print("Dataset saved")

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
			mags.append(np.transpose(np.abs(X)).astype('float32'))
			angles.append(np.angle(X).astype('float32'))
	else:
		print("COLA constraint not met, in func: utils.make_stft")
		exit()

	# print(len(mags),np.abs(mags[0].shape))
	return np.vstack(mags),angles

def get_stft_matrix(magnitudes, phases):
	return magnitudes * np.exp(1.j * phases)

def make_wav(mags, phases, overlap=overlap):
	a=[]
	for mag,phase in zip (mags,phases):
		mag=(mag.reshape(88,n_bins).swapaxes(1,0))
		# phase=np.transpose(phase.reshape(-1,n_bins))
		stft_matrix = get_stft_matrix(mag, phase)
		# print(stft_maxrix.shape)
		# for mat in stft_maxrix:
		# print(mat.shape)
		a.append(istft(stft_matrix,fs=sr, noverlap=overlap)[1])
		# print("one ",end="")
	# print(np.hstack(a).shape)
	return np.hstack(a)


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
	pickle.dump(dataset_stft, open(wavs_dir+"/dataset_stft.pickle", "wb"),pickle.HIGHEST_PROTOCOL)
	print("Dataset saved")

def multi_stft(mat,key):
	phase,angle=make_stft(mat)
	print(key)
	return [key,phase,angle]

def save_diff_stft(wavs_dir,dataset,index=0):
	# output = mp.Queue()
	mix,voc,acc,dru,bas,oth=dataset['mixture'],dataset['vocals'],dataset['accompaniment'],dataset['drums'],dataset['bass'],dataset['other']
	dataset_stft={}
	print('starting stft')
	
	keylist=list(dataset.keys())

	pool = mp.Pool(processes=6)
	results=[pool.apply(multi_stft,args=(mat,key)) for mat,key in zip ([dataset[keyl] for keyl in keylist],keylist)]

	print("out of the wormhole!")

	dataset_stft={}
	for result in results:
		dataset_stft[result[0]]=result[1]
		dataset_stft[result[0]+"angle"]=result[2]


	print("Saving dataset")
	pickle.dump(dataset_stft, open(wavs_dir+"/dataset_stft_"+str(index)+".pickle", "wb"),pickle.HIGHEST_PROTOCOL)
	print(" saved")


def read(dir_list,index):
	data=save_as_wav(dir_list)
	print(index)
	save_diff_stft(wavs_dir,data,index)
	return index
def read_mix_voc_acc(wavs_dir=wavs_dir,limit=49):
	mixl=[]
	vocl=[]
	accl=[]
	for index in range(limit[0],limit[1]-1,5):
		print("\rGetting Data: {0:.2f}%  ".format(index), end="")
		mix,voc,acc=read_data(wavs_dir+"/dataset_stft_"+str(index)+".pickle")
		mixl.append(mix)
		vocl.append(voc)
		accl.append(acc)
	zeros=np.zeros((1,n_bins))
	mixl=np.vstack(mixl)
	vocl=np.vstack(vocl)
	accl=np.vstack(accl)
	if len(mixl)%4 is not 0:
		rem=4-len(mixl)%4
		padding=np.repeat(zeros,rem,axis=0)
		print(padding.shape)
		mixl=np.vstack(mixl,padding)
	vocl=np.vstack(vocl)
	if len(vocl)%4 is not 0:
		rem=4-len(vocl)%4
		padding=np.repeat(zeros,rem,axis=0)
		print(padding.shape)
		vocl=np.vstack(vocl,padding)
	accl=np.vstack(accl)
	if len(accl)%4 is not 0:
		rem=4-len(accl)%4
		padding=np.repeat(zeros,rem,axis=0)
		print(padding.shape)
		accl=np.vstack(accl,padding)
	return mixl,vocl,accl

if __name__ == '__main__':
	dir_list=os.listdir(os.path.join(wavs_dir,'train'))

	# pool=mp.Pool(processes=20)

	results=[(read(dir_list[sub_list:sub_list+5],sub_list)) for sub_list in range(95,len(dir_list)-4,5)]
	# output = [p.get() for p in results]
	print(results)

	print("Ta-da!")