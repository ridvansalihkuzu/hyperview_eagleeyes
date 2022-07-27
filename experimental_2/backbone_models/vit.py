import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
      super(Patches, self).__init__()
      self.patch_size = patch_size
      self.num_patches = num_patches

    def call(self, images):
      batch_size = tf.shape(images)[0]
      patches = tf.image.extract_patches(
          images=images,
          sizes=[1, self.patch_size, self.patch_size, 1],
          strides=[1, self.patch_size, self.patch_size, 1],
          rates=[1, 1, 1, 1],
          padding="VALID",
      )
      patch_dims = patches.shape[-1]
      patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
      return patches

class PositionalEncoding(layers.Layer):
  def __init__(self, num_patches, embedding_dim):
    super(PositionalEncoding, self).__init__()
    self.num_patches = num_patches
    self.embedding_dim = embedding_dim  # Number of hidden units.
    self.positions = tf.range(start=0, limit=self.num_patches, delta=1)
    self.emb = layers.Embedding(input_dim=self.num_patches, output_dim=self.embedding_dim)

  def call(self, x):
    position_embedding = self.emb(self.positions)
    x = x + position_embedding
    return x

class Transformer(layers.Layer):
  def __init__(self, embedding_dim,drop_rate):
    super(Transformer, self).__init__()
    self.hidden_units=[embedding_dim, embedding_dim]
    self.dropout_rate=drop_rate
    self.norm1 = layers.LayerNormalization(epsilon=1e-5)
    self.norm2 = layers.LayerNormalization(epsilon=1e-5)
    self.attention = layers.MultiHeadAttention(num_heads=2, key_dim=128, dropout=self.dropout_rate)
    l = []
    for units in self.hidden_units:
      l.append(layers.Dense(units, activation=tf.nn.gelu))
      l.append(layers.Dropout(self.dropout_rate))
    self.mlp = keras.models.Sequential(l)

  def call(self, x):
    x = self.norm1(x)
    attention_output = self.attention(x, x)
    x1 = x + attention_output

    x = self.norm2(x1)
    x = self.mlp(x)
    return x + x1


class ViT(keras.models.Model):
    def __init__(self, input_shape, include_top=True,classifier_activation=None,num_classes=1000,
                 patch_size = 8, drop_rate=0.2, blocks=3, embedding_dim=256):

        num_patches = (input_shape[-2] // patch_size) ** 2

        encoder = keras.models.Sequential([
            Patches(patch_size, num_patches),
            layers.Dense(embedding_dim),
            PositionalEncoding(num_patches, embedding_dim)], name="encoder")

        transformers = keras.models.Sequential([Transformer(embedding_dim,drop_rate) for i in range(blocks)],
                                                    name="transformers")

        head = keras.models.Sequential([layers.GlobalAveragePooling1D(),
                                             layers.Dropout(rate=drop_rate),
                                             layers.Dense(num_classes,activation=classifier_activation)], name="classifier")

        inp = tf.keras.layers.Input(shape=input_shape)
        out1 = transformers(encoder(inp))
        out2 = head(out1)
        if include_top:
            super(ViT, self).__init__(inputs=inp, outputs=out2)
        else:
            super(ViT, self).__init__(inputs=inp, outputs=out1)

