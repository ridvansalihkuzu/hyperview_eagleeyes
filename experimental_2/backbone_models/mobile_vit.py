import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class InvertedRes(layers.Layer):
    def __init__(self, expand_channels, output_channels, strides=1):
        super().__init__(name='inverted_res')
        self.output_channels = output_channels
        self.strides = strides
        self.expand = keras.Sequential([
            layers.Conv2D(expand_channels, 1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish')
        ])
        self.dw_conv = keras.Sequential([
            layers.DepthwiseConv2D(3, strides=strides, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish')
        ])
        self.pw_conv = keras.Sequential([
            layers.Conv2D(output_channels, 1, padding="same", use_bias=False),
            layers.BatchNormalization(),
        ])

    def call(self, x):
        o = self.expand(x)
        o = self.dw_conv(o)
        o = self.pw_conv(o)
        if self.strides == 1 and o.shape[-1] == self.output_channels:
            return o + x
        return o

class FullyConnected(layers.Layer):
  def __init__(self, hidden_units, dropout_rate):
    super().__init__(name='fully_connected')
    l = []
    for units in hidden_units:
      l.append(layers.Dense(units, activation=tf.nn.swish))
      l.append(layers.Dropout(dropout_rate))
    self.mlp = keras.Sequential(l)

  def call(self, x):
    return self.mlp(x)


class Transformer(layers.Layer):
    def __init__(self, projection_dim, heads=2):
        super().__init__(name='transformer')
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=projection_dim, dropout=0.1)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.mlp = FullyConnected([input_shape[-1] * 2, input_shape[-1]], dropout_rate=0.1)

    def call(self, x):
        x1 = self.norm1(x)
        att = self.attention(x1, x1)
        x2 = x + att
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        return x3 + x2

class MobileVitBlock(layers.Layer):
  def __init__(self, num_blocks, projection_dim,patch_size=4, strides=1):
    super().__init__(name='mobile_vit_block')
    self.patch_size=patch_size
    self.projection_dim = projection_dim
    self.conv_local = keras.Sequential([
                                           layers.Conv2D(projection_dim, 3, padding="same", strides=strides, activation=tf.nn.swish),
                                           layers.Conv2D(projection_dim, 1, padding="same", strides=strides, activation=tf.nn.swish),
                                           ])
    self.transformers = keras.Sequential([Transformer(projection_dim, heads=2) for i in range(num_blocks)])
    self.conv_folded = layers.Conv2D(projection_dim, 1, padding="same", strides=strides, activation=tf.nn.swish)
    self.conv_local_global = layers.Conv2D(projection_dim, 3, padding="same", strides=strides, activation=tf.nn.swish)

  def build(self, input_shape):
    num_patches = int((input_shape[1] * input_shape[2]) / self.patch_size)
    self.unfold = layers.Reshape((self.patch_size, num_patches, self.projection_dim))
    self.fold = layers.Reshape((input_shape[1], input_shape[2], self.projection_dim))

  def call(self, x):
    local_features = self.conv_local(x)
    patches = self.unfold(local_features)
    global_features = self.transformers(patches)
    folded_features = self.fold(global_features)
    folded_features = self.conv_folded(folded_features)
    local_global_features = tf.concat([x, folded_features], axis=-1)
    local_global_features = self.conv_local_global(local_global_features)
    return local_global_features


class MobileVit(keras.Model):
    def __init__(self, input_shape, include_top=True,classifier_activation=None,
                                                        num_classes=1000,expansion_ratio = 2.0):
        inp=tf.keras.layers.Input(shape=input_shape)
        features = keras.models.Sequential([
                                                 layers.Conv2D(16, 3, padding="same", strides=(2, 2),
                                                               activation=tf.nn.swish),
                                                 InvertedRes(16 * expansion_ratio, 16, strides=1),
                                                 InvertedRes(16 * expansion_ratio, 24, strides=2),
                                                 InvertedRes(24 * expansion_ratio, 24, strides=1),
                                                 InvertedRes(24 * expansion_ratio, 24, strides=1),
                                                 InvertedRes(24 * expansion_ratio, 48, strides=2),
                                                 MobileVitBlock(2, 64, strides=1),
                                                 InvertedRes(64 * expansion_ratio, 64, strides=2),
                                                 MobileVitBlock(4, 80, strides=1),
                                                 InvertedRes(80 * expansion_ratio, 80, strides=2),
                                                 MobileVitBlock(3, 96, strides=1),
                                                 layers.Conv2D(320, 1, padding="same", strides=(1, 1),
                                                               activation=tf.nn.swish)
                                                 ], name="features")

        head = keras.models.Sequential([layers.GlobalAvgPool2D(),
                                             layers.Dense(num_classes, activation=classifier_activation)
                                             ], name="logits")
        out1 = features(inp)
        out2 = head(out1)

        if include_top:
            super(MobileVit, self).__init__(inputs=inp, outputs=out2)
        else:
            super(MobileVit, self).__init__(inputs=inp, outputs=out1)



class MobileVitC(keras.Model):
    def __init__(self, input_shape, include_top=True,classifier_activation=None,
                                                        num_classes=1000,expansion_ratio = 2.0):
        inp=tf.keras.layers.Input(shape=input_shape)
        features = keras.models.Sequential([
                                                 layers.Conv2D(16, 16, padding="same", strides=(2, 2),
                                                               activation=tf.nn.swish),
                                                 InvertedRes(16 * expansion_ratio, 16, strides=1),
                                                 InvertedRes(16 * expansion_ratio, 24, strides=2),
                                                 InvertedRes(24 * expansion_ratio, 24, strides=1),
                                                 InvertedRes(24 * expansion_ratio, 24, strides=1),
                                                 InvertedRes(24 * expansion_ratio, 48, strides=2),
                                                 MobileVitBlock(2, 64, strides=1),
                                                 InvertedRes(64 * expansion_ratio, 64, strides=2),
                                                 MobileVitBlock(4, 80, strides=1),
                                                 InvertedRes(80 * expansion_ratio, 80, strides=2),
                                                 MobileVitBlock(3, 96, strides=1),
                                                 layers.Conv2D(320, 1, padding="same", strides=(1, 1),
                                                               activation=tf.nn.swish)
                                                 ], name="features")



        head = keras.models.Sequential([layers.GlobalAvgPool2D(),
                                             layers.Dense(num_classes, activation=classifier_activation)
                                             ], name="logits")
        out1 = features(inp)
        out2 = head(out1)

        if include_top:
            super(MobileVitC, self).__init__(inputs=inp, outputs=out2)
        else:
            super(MobileVitC, self).__init__(inputs=inp, outputs=out1)

