# from librosa.output import write_wav
import numpy as np
from keras.models import load_model
from os import environ

import model
from TimeFreqMasking import TimeFreqMasking
from model import penalized_loss
from utils import make_stft,make_wav
from configuration import seq_len,n_bins

environ['TF_CPP_MIN_LOG_LEVEL']='1'

def separate(track,model='tempmodel.h5'):
	
	testcaseL,angleL=make_stft( [track.audio[:,0]])
	testcaseR,angleR=make_stft( [track.audio[:,1]])
	# print(track.name)
	# print(testcaseR.shape)
	stft_estimates={
	'musicL':[],
	'vocalL':[],
	'musicR':[],
	'vocalR':[]
	}
	testmodel=load_model(model,custom_objects={'TimeFreqMasking':TimeFreqMasking})

	stft_estimates['musicL'],stft_estimates['vocalL']=testmodel.predict(testcaseL.reshape(-1,seq_len,n_bins))
	stft_estimates['musicR'],stft_estimates['vocalR']=testmodel.predict(testcaseR.reshape(-1,seq_len,n_bins)) 

	# for testcase in testcaseL.reshape(-1,1,seq_len,n_bins):
	# 	# print(testcase.shape)
		
	# 	stft_estimates['musicL'].append(music_predL)
	# 	stft_estimates['vocalL'].append(vocal_predL)

	# for testcase in testcaseR.reshape(-1,1,seq_len,n_bins):
		
		
	# 	stft_estimates['musicR'].append(music_predR)
	# 	stft_estimates['vocalR'].append(vocal_predR)

	for key in stft_estimates.keys():
		stft_estimates[key]=np.array(stft_estimates[key]).reshape(-1,88,n_bins)

	vocals=np.stack([make_wav(stft_estimates['vocalL'],angleL),make_wav(stft_estimates ['vocalR'],angleR)],axis=1)
	music=np.stack([make_wav(stft_estimates['musicL'],angleL),make_wav(stft_estimates ['musicR'],angleR)],axis=1)


	# print(track.audio.shape)
	# print(np.pad(vocals,((0,(track.audio.shape[0]-vocals.shape[0])),(0,0)),'constant',constant_values=((0,0),(0, 0))).shape)
	# print(music.shape)

	# palceholder=np.ones((10,1))
	return {
		'vocals': np.pad(vocals,((0,np.abs(track.audio.shape[0]-vocals.shape[0])),(0,0)),'constant',constant_values=((0,0),(0, 0))),
		'accompaniment': np.pad(music,((0,np.abs(track.audio.shape[0]-music.shape[0])),(0,0)),'constant',constant_values=((0,0),(0, 0)))
		# 'bass': track.targets['bass'].audio,
		# 'drums': track.targets['drums'].audio,
		# 'other':  track.targets['other'].audio,
	}
'''
def IBM(track, alpha=1, theta=0.5):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal binary mask.
    the mix is send to some source if the spectrogram of that source over that
    of the mix is greater than theta, when the spectrograms are take as
    magnitude of STFT raised to the power alpha. Typical parameters involve a
    ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)
    """

    # parameters for STFT
    nfft = 2048

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # perform separtion
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():

        # compute STFT of target source
        Yj = stft(source.audio.T, nperseg=nfft)[-1]

        # Create Binary Mask
        Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
        Mask[np.where(Mask >= theta)] = 1
        Mask[np.where(Mask < theta)] = 0

        # multiply mask
        Yj = np.multiply(X, Mask)

        # inverte to time domain and set same length as original mixture
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        accompaniment_source += target_estimate

    # set accompaniment source
    estimates['accompaniment'] = accompaniment_source

    return estimates
    '''