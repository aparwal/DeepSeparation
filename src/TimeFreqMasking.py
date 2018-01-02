#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH -o monaural.out
#SBATCH -t 1-12:20:00
#SBATCH --gres=gpu:2
#SBATCH --mem=56000

# TimeFreqMasking.py 
# Anand Parwal
# Deep Learning for Advanced Robot Perception

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import RepeatVector
import numpy as np
from configuration import seq_len
### layer for keras 2.0 

class TimeFreqMasking(Layer):

    def __init__(self, output_dim=93, **kwargs):
        self.output_dim = output_dim
        super(TimeFreqMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel', 
                                      # shape=(input_shape[1], self.output_dim),
                                      # initializer='uniform',
                                      # trainable=True)

        super(TimeFreqMasking, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        y_hat_self=x[0]
        y_hat_other=x[1]
        x_mixed=x[2]

        
        m=( y_hat_self / (y_hat_self + y_hat_other + np.finfo(float).eps))
        mask=RepeatVector((seq_len))(m)

        # print(x_mixed.shape,m.shape,mask.shape)
        # print('Inside masking',m.shape)
        y_tilde_self =mask*x_mixed
        # print('Inside masking',y_tilde_self.shape)

        return y_tilde_self

    def compute_output_shape(self, input_shape):
        return (input_shape[2][0],input_shape[2][1],input_shape[2][2])

#################################################################################################
# ####################################################################################
# ### layer for keras 1.x 
# class TimeFreqMasking(Layer):
#   def __init__(self, output_dim=(ModelConfig.SEQ_LEN,513), **kwargs):
#       self.output_dim = output_dim
#       super(TimeFreqMasking, self).__init__(**kwargs)

#   def build(self, input_shape):
#       # Create a trainable weight variable for this layer.
#       # self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
#       #                        initializer='uniform',
#       #                        trainable=False)
#       super(TimeFreqMasking, self).build(input_shape)  # Be sure to call this somewhere!

#   def call(self, x, mask=None):
#       y_hat_self=x[0]
#       y_hat_other=x[1]
#       x_mixed=x[2]

#       y_tilde_self = y_hat_self / (y_hat_self + y_hat_other + np.finfo(float).eps)*x_mixed

#       return y_tilde_self

#   def get_output_shape_for(self, input_shape):
#       return (input_shape[0][0], self.output_dim[0],self.output_dim[1])
# ####################################################################################
