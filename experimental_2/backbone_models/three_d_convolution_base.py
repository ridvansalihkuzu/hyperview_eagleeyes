from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

#### LAYERS / HELPER FUNCS
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras import backend as K



class ThreeDCNN(tf.keras.Model):
    def __init__(self, input, n_class=2):

        ## convolutional layers

        conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input)
        conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

        ## add max pooling to obtain the most imformatic features
        pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

        conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
        conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
        pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

        ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
        pooling_layer2 = BatchNormalization()(pooling_layer2)
        flatten_layer = Flatten()(pooling_layer2)

        ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
        ## add dropouts to avoid overfitting / perform regularization
        dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=n_class, activation='linear')(dense_layer2)



        super(ThreeDCNN, self).__init__(inputs=input, outputs=output_layer,name='total')



