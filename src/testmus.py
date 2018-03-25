import musdb
import functools
from configuration import data_dir,estimates_dir,wavs_dir,result_dir
from scipy.io import wavfile
from separate import separate

mus = musdb.DB(root_dir=data_dir)
global alpha
model='tempmodel.h5'
subsets="test"

# mus.test(separate)
# exit()

mus.run(
	functools.partial(separate,model=model),
	estimates_dir=result_dir)
	# subsets=subsets)

 
#for decoding stem and saving in estimates_dir

# def decode(track):

# 	wavfile.write(wavs_dir+"/"+subsets+"/"+track.name+"/mixture.wav",track.rate,track.audio )
# 	return {
# 		'vocals': track.targets['vocals'].audio,
# 		'bass': track.targets['bass'].audio,
# 		'accompaniment': track.targets['accompaniment'].audio,
# 		'drums': track.targets['drums'].audio,
# 		'other':  track.targets['other'].audio,
# 	}

# mus.run(decode,estimates_dir=wavs_dir,subsets=subsets)