
data_dir="../data/MUS-STEMS-SAMPLE"
estimates_dir="../estimates"
wavs_dir="../data/wavs"
sr=44100;

time_len=1;
perseg=1024;
overlap = perseg // 2;
n_bins=perseg//2+1

seq_len = 4;
stateful=False
regular_const=[0.5,0.5]
loss_weights=[1, 1]

# Train
class TrainConfig:
    epochs=10
    batch_size=512
    validation_split=0.2
    # SECONDS = 120 ######## needs to be a multiple of 8 for seq_len=4
    WRITE_SAMPLE=False
    # CASE="seq-{}-dft-{}-hop-{}".format(ModelConfig.SEQ_LEN,ModelConfig.DFT,(ModelConfig.HOP*100/ModelConfig.DFT))
    # WRITE_PATH="result_train/"+CASE



    # MAXLEN=SECONDS*ModelConfig.SR
    # N_FRAMES=((SECONDS*ModelConfig.SR + ModelConfig.HOP - 1) // ModelConfig.HOP)


'''# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# import tensorflow as tf

# Model
class ModelConfig:
    SR = 16000
    DFT = 1024
    HOP = int(DFT / 4)
    SEQ_LEN = 4
    N_BINS=int(DFT/2+1)
    loss_weights=[1., 1.]
    regular_const=[0,0]
    stateful=True


# Train
class TrainConfig:
    epochs=30
    batch_size=1
    validation_split=0.2
    SECONDS = 120 ######## needs to be a multiple of 8 for seq_len=4
    WRITE_SAMPLE=True
    CASE="seq-{}-dft-{}-hop-{}".format(ModelConfig.SEQ_LEN,ModelConfig.DFT,(ModelConfig.HOP*100/ModelConfig.DFT))
    WRITE_PATH="result_train/"+CASE



    MAXLEN=SECONDS*ModelConfig.SR
    N_FRAMES=((SECONDS*ModelConfig.SR + ModelConfig.HOP - 1) // ModelConfig.HOP)

    # CASE = str(ModelConfig.SEQ_LEN) + 'frames_mir1k'
    # CKPT_PATH = 'checkpoints/' + CASE
    # GRAPH_PATH = 'graphs/' + CASE + '/train'
    # # DATA_PATH = 'dataset/train/ikala'
    # DATA_PATH = 'dataset/mir-1k/Wavfile'
    # LR = 0.0001
    # FINAL_STEP = 100000
    # CKPT_STEP = 500
    # NUM_WAVFILE = 1
    
    # RE_TRAIN = False

    # session_conf = tf.ConfigProto(
    #     device_count={'CPU': 1, 'GPU': 0},
        # gpu_options=tf.GPUOptions(
        #     allow_growth=True,
        #     per_process_gpu_memory_fraction=0.25
        # ),
    # )


# TODO seperating model and case
# TODO config for each case
# Eval
class EvalConfig:
    # CASE = '1frame'
    # CASE = '4-frames-masking-layer'
    CASE = str(ModelConfig.SEQ_LEN) + 'frames_mir1k'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    # DATA_PATH = 'dataset/eval/kpop'
    DATA_PATH = 'dataset/mir-1k/Wavfile'
    DATA_PATH="../MIR-1K/Wavfile"
    # DATA_PATH = 'dataset/ikala'
    GRIFFIN_LIM = False
    GRIFFIN_LIM_ITER = 1000
    NUM_EVAL = 1
    SECONDS = 10
    RE_EVAL = True
    EVAL_METRIC = True
    WRITE_RESULT = True
    RESULT_PATH = 'results/' + CASE 
    # session_conf = tf.ConfigProto(
    #     device_count={'CPU': 1, 'GPU': 0},
    #     # gpu_options=tf.GPUOptions(allow_growth=True),
    #     # log_device_placement=False
    # )
'''