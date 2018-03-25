# DeepSeparation
Keras Implementation and Experiments with Joint Optimization of Masks and Deep Recurrent Neural Networks for Source Separation

Using a custom designed keras layer for time frequency masking 

> Project under development

## Dependencies

* python 3.x,
* keras2.x 
* SciPy 
* [musdb](https://github.com/sigsep/sigsep-mus-db)

## Usage

In `configuration.py`, set `data_dir` to folder containing test files and `results_dir` to output folder and run `test.py`

### Parameters tested

Input shape:
This is dependent on the sampleing rate, for DSD100 rate of 44.1 kHz, one second of audio, scipy fft by default will make 513 bins. Sequence length of 4 makes the input shape [N,513,4]

Number of LSTM layers:
3,2,1

Uints per layer:
256,512

Activation funciton:
ReLu, tanh

L2 regularization on recurrent layers:
0.0 1.0

Batch normalization:
yes and no

Loss = mse + [reg const]discriminative
reg const : 0,0.5,1

### Callbacks

Writing on tensorboard, early stopping and reduce learning rate on plateau

## Reference
1. P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "[Joint Optimization of Masks and Deep Recurrent Neural Networks for Monaural Source Separation](http://posenhuang.github.io/papers/Joint_Optimization_of_Masks_and_Deep%20Recurrent_Neural_Networks_for_Monaural_Source_Separation_TASLP2015.pdf)", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 12, pp. 2136–2147, Dec. 2015
