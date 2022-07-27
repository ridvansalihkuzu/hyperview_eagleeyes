"""DenseNet 1DCNN in Tensorflow-Keras
Reference: Densely Connected Convolutional Networks [https://arxiv.org/abs/1608.06993]
"""

import tensorflow as tf


def Conv_1D_Block(x, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="same")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    return pool


def conv_block(x, num_filters, bottleneck=True):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    if bottleneck:
        num_filters_bottleneck = num_filters * 4
        x = Conv_1D_Block(x, num_filters_bottleneck, 1, 1)

    out = Conv_1D_Block(x, num_filters, 3, 1)

    return out


def dense_block(x, num_filters, num_layers, bottleneck=True):
    for i in range(num_layers):
        cb = conv_block(x, num_filters, bottleneck=bottleneck)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)

    return x


def transition_block(inputs, num_filters):
    x = Conv_1D_Block(inputs, num_filters, 1, 2)
    if x.shape[1] <= 2:
        x = tf.keras.layers.AveragePooling1D(pool_size=1, strides=2, padding="same")(x)
    else:
        x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(x)

    return x


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)

    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)

    return out


class DenseNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, bottleneck=True):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten(name='flatten')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def DenseNet121(self,name='total'):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 24, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 16, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs,name='total')

        return model

    def DenseNet161(self,name='total'):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters * 2)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters * 4)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 36, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters * 8)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 24, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs,name='total')

        return model

    def DenseNet169(self,name='total'):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters * 2)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters * 4)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 32, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters * 8)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 32, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs,name='total')

        return model

    def DenseNet201(self,name='total'):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 48, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 32, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs,name='total')

        return model

    def DenseNet264(self,name='total'):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters * 2)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters * 4)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 64, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters * 8)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 48, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs,name='total')

        return model