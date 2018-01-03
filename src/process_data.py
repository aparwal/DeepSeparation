import musdb
import functools
from configuration import data_dir,estimates_dir,wavs_dir
from scipy.io import wavfile
from separate import separate
import os

mus = musdb.DB(root_dir=data_dir)
global alpha
model='tempmodel.h5'
subsets="train"

# mus.test(separate)
# exit()

# mus.run(
# 	functools.partial(separate,model=model),
# 	estimates_dir=estimates_dir,
# 	subsets="train")


#for decoding stem and saving in estimates_dir

def decode(track):
	directory=wavs_dir+"/"+subsets+"/"+track.name
	if not os.path.exists(directory):
    		os.makedirs(directory)
	wavfile.write(directory+"/mixture.wav",track.rate,track.audio )
	return {
		'vocals': track.targets['vocals'].audio,
		'bass': track.targets['bass'].audio,
		'accompaniment': track.targets['accompaniment'].audio,
		'drums': track.targets['drums'].audio,
		'other':  track.targets['other'].audio,
	}

mus.run(decode,estimates_dir=wavs_dir,subsets="train",parallel=True, cpus=20)