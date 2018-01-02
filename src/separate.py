# from librosa.output import write_wav
import numpy as np
from keras.models import load_model
from TimeFreqMasking import TimeFreqMasking
from utils import make_stft
from configuration import seq_len,n_bins


def separate(track,alpha=0):
	f='tempmodel.h5'
	testcaseL=make_stft( [track.audio[:,0]])
	testcaseR=make_stft( [track.audio[:,1]])
	print(track.name)
	print(testcaseR.shape)
	musicL=[]
	vocalL=[]
	musicR=[]
	vocalR=[]

	for testcase in testcaseL.reshape(-1,seq_len,n_bins):
		testmodel=load_model(f,custom_objects={'TimeFreqMasking':TimeFreqMasking})
		music_predL,vocal_predL=testmodel.predict(testcaseL)
		musicL.append(music_predL)
		vocalL.append(vocal_predL)

	for testcase in testcaseR.reshape(-1,seq_len,n_bins):
		testmodel=load_model(f,custom_objects={'TimeFreqMasking':TimeFreqMasking})
		music_predR,vocal_predR=testmodel.predict(testcaseR)
		musicR.append(music_predR)
		vocalR.append(vocal_predR)


	vocals=np.hstack([np.array(vocalL),np.array(vocalR)])
	music=np.hstack([np.array(musicL),np.array(musicR)])
	print(vocals.shape)
	print(music.shape)

	# palceholder=np.ones((10,1))
	return {
		'vocals': vocals,
		'accompaniment': music
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