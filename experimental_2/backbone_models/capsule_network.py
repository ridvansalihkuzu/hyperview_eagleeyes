'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.
This file contains the network definitions for the various capsule network architectures.
'''

from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import math
from tensorflow.keras.layers import *
K.set_image_data_format('channels_last')

#### LAYERS / HELPER FUNCS
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

class Length(layers.Layer):
    """
    Final layer that computes the segmentation mask of some capsule grid with one capsule of some arbitrary length
    ** Computes length of output vectors because they represent the class probabilities
    input_shape: [None, h, w, num_capsules = 1, num_dims]
    output_shape: [None, h, w, 1]
    """
    def __init__(self, num_classes, seg=True, **kwargs):
        super(Length, self).__init__(**kwargs)
        # deciding the output channels for output seg
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg

    def call(self, inputs, **kwargs):
        if inputs.get_shape().ndims == 5:
            assert inputs.get_shape()[-2] == 1, 'Error: Must have num_capsules = 1 going into Length'
            inputs = K.squeeze(inputs, axis=-2)
        # gets final result by getting the lengths of the final capsules for each pixel
        #### NOTE: USING `tf_norm` IS RISKY
        return K.expand_dims(tf.norm(inputs, axis= -1), axis= -1)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 5:
            input_shape = input_shape[0:-2] + input_shape[-1:]
        if self.seg:
            return input_shape[:-1] + (self.num_classes,)
        else:
            return input_shape[:-1]

    def get_config(self):
        config = {'num_classes': self.num_classes, 'seg': self.seg}
        base_config = super(Length, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Mask(layers.Layer):
    """
    Masks input tensors with either resize_mask or the capsules with max dim
    """
    def __init__(self, resize_masks=False, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.resize_masks = resize_masks

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            input, mask = inputs
            _, hei, wid, _, _ = input.get_shape()
            if self.resize_masks:
                mask = tf.image.resize_bicubic(mask, (hei.value, wid.value))
            mask = K.expand_dims(mask, -1)
            if input.get_shape().ndims == 3:
                masked = K.batch_flatten(mask * input)
            else:
                masked = mask * input

        else:
            if inputs.get_shape().ndims == 3:
                x = K.sqrt(K.sum(K.square(inputs), -1))
                mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
                masked = K.batch_flatten(K.expand_dims(mask, -1) * inputs)
            else:
                masked = inputs

        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            if len(input_shape[0]) == 3:
                return tuple([None, input_shape[0][1] * input_shape[0][2]])
            else:
                return input_shape[0]
        else:  # no true label provided
            if len(input_shape) == 3:
                return tuple([None, input_shape[1] * input_shape[2]])
            else:
                return input_shape

    def get_config(self):
        config = {'resize_masks': self.resize_masks}
        base_config = super(Mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                    num_routing):
    """
    param votes [batch_size, i_num_caps, conv_height, conv_width, num_caps, num_atoms]
    param biases_replicated [conv_height, conv_width, num_caps, num_atoms]
    param logit_shape [batch_size, i_num_caps, conv_height, conv_width, num_caps]
    returns:
    the final tensor from activations [conv_height, conv_width, num_atoms, num_caps]
    """
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    # transposes the 6D matrix to [num_atoms, batch_size, i_num_caps, conv_height, conv_width, n_caps]
    votes_trans = tf.transpose(votes, votes_t_shape)
    _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """
        Routing while loop from the paper: https://arxiv.org/pdf/1804.04241.pdf
        param i: tf.constant representing the iteration integer (automatically updated) for the condition in the while loop
        param logits: routing weights [batch_size, i_num_caps, conv_height, conv_width, num_caps]
        param activations: tensor array of resultant vectors
        returns:
        i + 1: to symbolize that one iteration has passed
        logits: updated routing weights
        activations: writes to the original TensorArray;  [conv_height, conv_width, num_atoms, num_caps]
        """

        # route: [batch, input_dim, output_dim, ...]

        # 4: routing softmax
        route = tf.nn.sigmoid(logits)
        # 5: summing the dot products between the routing coefficients and u-hat (votes)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        # 6: squashing the summed dot products
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        # 7: routing weights updating
        act_3d = K.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=-1)
        logits += distances
        return (i + 1, logits, activations)

    # loop vars
    activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False) # stores all the resultant vectors of the parent capsule for each iteration
    logits = tf.random.uniform(logit_shape, 0,1) # initializes the routing weights to 0
    i = tf.constant(0, dtype=tf.int32)

    # the while loop
    _, logits, activations = tf.while_loop(
      lambda i, logits, activations: i < num_routing,
      _body,
      loop_vars=[i, logits, activations],
      swap_memory=True)
    # returns the final resultant vector
    return K.cast(activations.read(num_routing - 1), dtype='float32')

####### uses dangerous norm, maybe change to a safe norm?
def _squash2(input_tensor):
    """
    Activation that squashes capsules to lengths between 0 and 1, where more significant capsules are closer to a lenght of 1 and less significant capsules are pushed towards 0
    * computes the raw norm; most likely need to change for a safer norm
    """

    return  K.elu(input_tensor)

def _squash(input_tensor):
    """
    Activation that squashes capsules to lengths between 0 and 1, where more significant capsules are closer to a lenght of 1 and less significant capsules are pushed towards 0
    * computes the raw norm; most likely need to change for a safer norm
    """
    norm = tf.norm(input_tensor, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

class ConvCapsuleLayer(layers.Layer):
    """
    takes a 5D tensor [None, h, w, input_n_capsules, input_n_atoms]
    param kernel_size:
    param num_capsule:
    param num_atoms: length of each capsule vector
    param strides:
    param padding:
    param routings: number of locally constrained routing iterations
    param kernel_initializer:
    returns
    """

    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', routings=3,
                 kernel_initializer='glorot_uniform', **kwargs):
        super(ConvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        # shape (x, y, input_n_atoms, n_capsules*n_atoms)
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                        self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):

        # changed to [input_n_capsules, None, h,w, input_n_atoms]
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        # n_capsules multiplied with n_samples
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_num_atoms])

        input_tensor_reshaped.set_shape(
            (None, self.input_height, self.input_width, self.input_num_atoms))
        conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding, data_format='channels_last')

        #conv=ECA()

        # shape of the routing coefficients?
        votes_shape = K.shape(conv)  # shape [None, h, w, input_n_capsules]
        _, conv_height, conv_width, _ = conv.get_shape()

        # reshapes output to: [None, input_n_capsules h_votes,w_votes, n_capsules, input_n_atoms]
        votes = K.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                 self.num_capsule, self.num_atoms])
        votes.set_shape((None, self.input_num_capsule, conv_height, conv_width,
                         self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(
            self.b, [conv_height, conv_width, 1, 1])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.strides,
                dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ECA(tf.keras.layers.Layer):
    """ECA Conv layer.
    NOTE: This should be applied after a convolution operation.
    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C_2, H, W)
    Attributes:
        filters (int): number of channels of input
        eca_k_size (int): kernel size for the 1D ECA layer
    """

    def __init__(
            self,
            gamma=2,
            b=2,
            kernel=None,
            **kwargs):

        super(ECA, self).__init__()

        self.kwargs = kwargs
        self.b = b
        self.gamma=gamma
        self.kernel=kernel

    def build(self, input_shapes):

        if self.kernel==None:
            t = int(abs(math.log(input_shapes[-1], 2) + self.b)/self.gamma)
            k = t + 1 if t % 2 else t
        else:
            k=self.kernel

        self.eca_conv = Conv1D(
            filters=1,
            kernel_size=k,
            padding='same',
            use_bias=False)

    def get_config(self):
        config = super(ECA, self).get_config().copy()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):

        # (B, C, 1)
        attn = tf.math.reduce_mean(x, [-4,-3, -2])[:, :,tf.newaxis]
        # (B, C, 1)
        attn = self.eca_conv(attn)

        # (B, 1, 1, C)
        attn=tf.transpose(attn, [0, 2, 1])
        attn = tf.expand_dims(attn, [-3])
        attn = tf.expand_dims(attn, [-4])

        # (B, 1, 1, C)
        attn = tf.math.sigmoid(attn)

        return x * attn


class DeconvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, scaling=2, upsamp_type='subpix', padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(DeconvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.scaling = scaling
        self.upsamp_type = upsamp_type
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        if self.upsamp_type == 'subpix':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.input_num_atoms,
                                            self.num_capsule * self.num_atoms * self.scaling * self.scaling],
                                     initializer=self.kernel_initializer,
                                     name='W')
        elif self.upsamp_type == 'resize':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                     self.input_num_atoms, self.num_capsule * self.num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        elif self.upsamp_type == 'deconv':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.num_capsule * self.num_atoms, self.input_num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        else:
            raise NotImplementedError('Upsampling must be one of: "deconv", "resize", or "subpix"')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[1] * input_shape[0], self.input_height, self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))


        if self.upsamp_type == 'resize':
            upsamp = K.resize_images(input_tensor_reshaped, self.scaling, self.scaling, 'channels_last')
            outputs = K.conv2d(upsamp, kernel=self.W, strides=(1, 1), padding=self.padding, data_format='channels_last')
        elif self.upsamp_type == 'subpix':
            conv = K.conv2d(input_tensor_reshaped, kernel=self.W, strides=(1, 1), padding='same',
                            data_format='channels_last')
            outputs = tf.nn.depth_to_space(conv, self.scaling)
        else:
            batch_size = input_shape[1] * input_shape[0]

            # Infer the dynamic output shape:
            out_height = deconv_output_length(self.input_height,self.kernel_size,self.padding,stride=self.scaling)
            out_width = deconv_output_length(self.input_width, self.kernel_size, self.padding, stride=self.scaling)
            output_shape = (batch_size, out_height, out_width, self.num_capsule * self.num_atoms)

            outputs = K.conv2d_transpose(input_tensor_reshaped, self.W, output_shape, (self.scaling, self.scaling),
                                     padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(outputs)
        _, conv_height, conv_width, _ = outputs.get_shape()

        votes = K.reshape(outputs, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                 self.num_capsule, self.num_atoms])
        votes.set_shape((None, self.input_num_capsule, conv_height, conv_width,
                         self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [votes_shape[1], votes_shape[2], 1, 1])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        output_shape[1] = deconv_output_length(output_shape[1], self.kernel_size, self.padding, stride=self.scaling)
        output_shape[2] = deconv_output_length(output_shape[2], self.kernel_size, self.padding, stride=self.scaling)
        output_shape[3] = self.num_capsule
        output_shape[4] = self.num_atoms

        return tuple(output_shape)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'scaling': self.scaling,
            'padding': self.padding,
            'upsamp_type': self.upsamp_type,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(DeconvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.
  Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full", "causal"
      stride: integer.
      dilation: dilation rate, integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full', 'causal'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding in ['same', 'causal']:
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


def deconv_output_length(input_length,
                         filter_size,
                         padding,
                         output_padding=None,
                         stride=0,
                         dilation=1):
  """Determines output length of a transposed convolution given input length.
  Args:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"same"`, `"valid"`, `"full"`.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.
  Returns:
      The output length (integer).
  """
  assert padding in {'same', 'valid', 'full'}
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'valid':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'full':
      length = input_length * stride - (stride + filter_size - 2)
    elif padding == 'same':
      length = input_length * stride

  else:
    if padding == 'same':
      pad = filter_size // 2
    elif padding == 'valid':
      pad = 0
    elif padding == 'full':
      pad = filter_size - 1

    length = ((input_length - 1) * stride + filter_size - 2 * pad +
              output_padding)
  return length



def CapsNetR3(input_shape, n_class=4):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv3D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H, W, 1, C))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='subpix',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='subpix',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='subpix',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    #out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()

    def shared_decoder(mask_layer,name):
        recon_remove_dim = layers.Reshape((H, W, A))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='glorot_uniform',
                                activation=layers.LeakyReLU(), name='recon_1_{}'.format(name))(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='glorot_uniform',
                                activation=layers.LeakyReLU(), name='recon_2_{}'.format(name))(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='glorot_uniform',
                                  activation='sigmoid', name='out_recon_{}'.format(name))(recon_2)

        return out_recon

    o1=shared_decoder(seg_caps,1)
    eval_model = models.Model(inputs=x, outputs=o1, name="capsule_network")
    # manipulate model
    #noise = layers.Input(shape=((H, W, C, A)))
    #noised_seg_caps = layers.Add()([seg_caps, noise])
    #masked_noised_y = Mask()([noised_seg_caps, y])
    #manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return eval_model #manipulate_model




class CapsNetBasic(tf.keras.Model):
    def __init__(self, input, n_class=2):


        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(input)

        # Reshape layer to be 1 capsule x [filters] atoms
        _, H, W, C = conv1.get_shape()
        conv1_reshaped = layers.Reshape((H, W, 1, C))(conv1)

        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=64, strides=2, padding='same',
                                        routings=1, name='primarycaps')(conv1_reshaped)
        primary_caps = ECA()(primary_caps)
        # Layer 2: Convolutional Capsule
        conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=128, strides=1, padding='same',
                                        routings=1, name='conv_cap_2_1')(primary_caps)
        conv_cap_2_1 = ECA()(conv_cap_2_1)
        # Layer 2: Convolutional Capsule
        conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                        routings=3, name='conv_cap_2_2')(conv_cap_2_1)
        conv_cap_2_2 = ECA()(conv_cap_2_2)
        #conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
        #                                routings=3, name='conv_cap_2_2')(conv_cap_2_1)
        #conv_cap_2_2 = ECA()(conv_cap_2_2)
        # Layer 4: Convolutional Capsule: 1x1
        out_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='seg_caps')(conv_cap_2_2)
        out_caps = ECA()(out_caps)
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(out_caps)

        out_seg=Flatten()(out_seg)
        out_seg=Dense(n_class,activation='sigmoid')(out_seg)

        super(CapsNetBasic, self).__init__(inputs=input, outputs=out_seg)
