import tensorflow as tf
from tensorflow.keras.layers import *
from experimental_2.backbone_models.swin_transformer import SwinTransformer, SwinTransformer3
from experimental_2.backbone_models.mobile_vit import MobileVit,MobileVitC
from experimental_2.backbone_models.vit import ViT
from experimental_2.backbone_models.capsule_network import CapsNetBasic
from experimental_2.backbone_models.three_d_convolution_base import ThreeDCNN
from tensorflow.keras import activations
import math



class SpatioMultiChannellModel(tf.keras.Model):

    def __init__(self, model_type, built_type, input_shape, label_shape, pretrained):

        temporal_input = tf.keras.layers.Input(shape=input_shape)
        if built_type==1:
            # ANY MODEL x 50 : (for every 3 channel, but the models are shring the weights)
            # For all soil parameters, there is a single backbone model with multiple prediction head
            fet_out=SpatioMultiChannellModel._multi_channel_builder_1(model_type, pretrained, label_shape, temporal_input)

        elif built_type == 2:
            # ANY MODEL x 50 : (for every 3 channel, a new model has been built with indepedent weights)
            # For all soil parameters, there is a single backbone model with multiple prediction head
            fet_out = SpatioMultiChannellModel._multi_channel_builder_2(model_type, pretrained, label_shape,temporal_input)

        elif built_type==3:
            # ANY MODEL with channel dimensional reduction with convolution
            # For all soil parameters, there is a single backbone model with multiple prediction head
            fet_out = SpatioMultiChannellModel._multi_channel_builder_3(model_type, pretrained, label_shape,temporal_input)

        elif built_type==4:
            # ANY MODEL with channel dimensional reduction with convolution and channel attention
            # For all soil parameters, there is a single backbone model with multiple prediction head
            fet_out=SpatioMultiChannellModel._multi_channel_builder_4(model_type, pretrained, label_shape,temporal_input)

        elif built_type==5:
            # ANY MODEL with channel dimensional reduction with convolution and channel attention
            # For each soil parameter, there are different backbone models
            fet_out=SpatioMultiChannellModel._multi_channel_builder_5(model_type, pretrained, label_shape,temporal_input)

        elif built_type==6:
            # CAPSULE NET TRIAL
            # For all soil parameters, there is a single backbone model with multiple prediction head
            fet_out=SpatioMultiChannellModel._multi_channel_builder_6(model_type, pretrained, label_shape,temporal_input)

        elif built_type==7:
            # CAPSULE NET TRIAL with channel dimensional reduction with convolution
            # For all soil parameters, there is a single backbone model with multiple prediction head
            fet_out=SpatioMultiChannellModel._multi_channel_builder_7(model_type, pretrained, label_shape,temporal_input)

        elif built_type==8:
            # 3D CNN model
            # For all soil parameters, there is a single backbone model with multiple prediction head
            fet_out=SpatioMultiChannellModel._multi_channel_builder_8(model_type, pretrained, label_shape,temporal_input)

        [P_logit,K_logit,Mg_logit,pH_logit]=tf.unstack(fet_out, axis=-1)

        P_out = Activation(activation=activations.linear,name='P')(P_logit)
        K_out = Activation(activation=activations.linear, name='K')(K_logit)
        Mg_out = Activation(activation=activations.linear, name='Mg')(Mg_logit)
        pH_out = Activation(activation=activations.linear, name='pH')(pH_logit)


        super(SpatioMultiChannellModel, self).__init__(inputs=temporal_input, outputs=[fet_out, P_out, K_out, Mg_out, pH_out])

    @staticmethod
    def _multi_channel_builder_1(model_type,pretrained,label_shape, temporal_input):
        # ANY MODEL x 50 : (for every 3 channel, but the models are shring the weights)
        # For all soil parameters, there is a single backbone model with multiple prediction head

        t_shape = temporal_input.shape
        input_list = tf.split(temporal_input, num_or_size_splits=int(t_shape[-1] / 3), axis=-1)
        feature = tf.squeeze(tf.stack(input_list, axis=1), -4)

        backbone = BackboneModel(model_type, feature.shape[2:], pretrained)


        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(TimeDistributed(backbone, input_shape=feature.shape[1:]))
        multi_chanel_model.add(Flatten())
        multi_chanel_model.add(BatchNormalization())
        multi_chanel_model.add(Dense(label_shape, activation='sigmoid'))

        out=multi_chanel_model(feature)
        return out

    @staticmethod
    def _multi_channel_builder_2(model_type,pretrained,label_shape, temporal_input):
        # ANY MODEL x 50 : (for every 3 channel, a new model has been built with indepedent weights)
        # For all soil parameters, there is a single backbone model with multiple prediction head

        t_shape = temporal_input.shape
        input_list = tf.split(temporal_input, num_or_size_splits=int(t_shape[-1] / 3), axis=-1)
        out_list=[]
        for input in input_list:
            backbone = BackboneModel(model_type, input.shape[2:], pretrained)
            out_list.append(backbone(tf.squeeze(input, -4)))

        feature = tf.stack(out_list, axis=1)


        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(Flatten())
        multi_chanel_model.add(Dense(label_shape, activation='sigmoid'))

        out=multi_chanel_model(feature)
        return out

    @staticmethod
    def _multi_channel_builder_3(model_type, pretrained, label_shape, temporal_input):
        # ANY MODEL with channel dimensional reduction with convolution
        # For all soil parameters, there is a single backbone model with multiple prediction head

        input = tf.squeeze(temporal_input, -4)
        multi_chanel_model = tf.keras.Sequential()
        multi_chanel_model.add(Conv2D(filters=256, kernel_size=(1, 1)))
        out = multi_chanel_model(input)

        backbone = BackboneModel(model_type, out.shape[1:], pretrained)

        out = backbone(out)
        out = Layer(name='total')(out)

        return out

    @staticmethod
    def _multi_channel_builder_4(model_type, pretrained, label_shape, temporal_input):
        # ANY MODEL with channel dimensional reduction with convolution and channel attention
        # For all soil parameters, there is a single backbone model with multiple prediction head

        input = tf.squeeze(temporal_input, -4)
        multi_chanel_model = tf.keras.Sequential()  # for channel dimension reduction
        multi_chanel_model.add(Conv2D(filters=128, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=9, name='eca1'))  # for channel-wise attention
        multi_chanel_model.add(Conv2D(filters=16, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=3, name='eca2'))  # for channel-wise attention
        out = multi_chanel_model(input)

        backbone = BackboneModel(model_type, out.shape[1:], pretrained)

        out = backbone(out)
        out = Layer(name='total')(out)

        return out

    @staticmethod
    def _multi_channel_builder_5(model_type, pretrained, label_shape, temporal_input):
        # ANY MODEL with channel dimensional reduction with convolution and channel attention
        # For each soil parameter, there are different backbone models

        input = tf.squeeze(temporal_input, -4)
        # multi_chanel_model = tf.keras.Sequential()
        # multi_chanel_model.add()
        con0 = Conv2D(filters=256, kernel_size=(1, 1))
        eca0 = ECA(kernel=5, grad=0)
        eca1 = ECA(kernel=5, grad=1)
        eca2 = ECA(kernel=5, grad=2)

        fnet = FNetEncoder(256, 256)
        # multi_chanel_model.add(Conv2D(filters=3, kernel_size=(1, 1), activation='relu'))
        con1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')
        con2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')
        con3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')
        con4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')

        out = con0(input)
        out1 = con1(eca0(out))
        out2 = con2(eca1(out))
        out3 = con3(eca2(out))
        out4 = con4(fnet(out))

        backbone1 = BackboneModel(model_type, out1.shape[1:], pretrained, 4)
        backbone2 = BackboneModel(model_type, out2.shape[1:], pretrained, 4)
        backbone3 = BackboneModel(model_type, out3.shape[1:], pretrained, 4)
        backbone4 = BackboneModel(model_type, out4.shape[1:], pretrained, 4)

        out1 = backbone1(out1)
        out2 = backbone2(out2)
        out3 = backbone3(out3)
        out4 = backbone4(out4)

        out = tf.stack([out1, out2, out3, out4], axis=-1)
        out = tf.reduce_mean(out, axis=-1)

        out = Layer(name='total')(out)

        return out



    @staticmethod
    def _multi_channel_builder_6(model_type, pretrained, label_shape, temporal_input):
        # CAPSULE NET TRIAL
        # For all soil parameters, there is a single backbone model with multiple prediction head

        t_shape = temporal_input.shape
        input=tf.squeeze(temporal_input,-4)
        #input_list = tf.split(temporal_input, num_or_size_splits=int(t_shape[-1] / 3), axis=-1)
        feature=CapsNetBasic(input,label_shape)



        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(Flatten())
        multi_chanel_model.add(Dense(label_shape, activation='sigmoid'))

        out = multi_chanel_model(feature(input))
        return out

    @staticmethod
    def _multi_channel_builder_7(model_type, pretrained, label_shape, temporal_input):
        # CAPSULE NET TRIAL with channel dimensional reduction with convolution
        # For all soil parameters, there is a single backbone model with multiple prediction head

        input = tf.squeeze(temporal_input, -4)
        multi_chanel_model = tf.keras.Sequential()  # for channel dimension reduction
        multi_chanel_model.add(Conv2D(filters=128, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=9, name='eca1'))  # for channel-wise attention
        multi_chanel_model.add(Conv2D(filters=16, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=3, name='eca2'))  # for channel-wise attention
        out = multi_chanel_model(input)

        backbone = CapsNetBasic(out, label_shape)

        out = backbone(out)

        out = Layer(name='total')(out)

        return out

    @staticmethod
    def _multi_channel_builder_8(model_type, pretrained, label_shape, temporal_input):
        # 3D CNN model
        # For all soil parameters, there is a single backbone model with multiple prediction head

        inp=tf.transpose(temporal_input, (0, 4, 2, 3, 1))
        model=ThreeDCNN(inp,label_shape)
        return model(inp)



class BackboneModel(tf.keras.Model):
        def __init__(self, model_type, input_shape,pretrained,out_shape=4):

            inp = tf.keras.layers.Input(shape=input_shape)
            model=None
            weights = 'imagenet' if pretrained else None
            if model_type == 0:
                model = SwinTransformer( model_name='swin_tiny_224', num_classes=1000,include_top=False, pretrained=pretrained)

            if model_type == 1:
                model=SwinTransformer3(input_shape=input_shape,model_name='swin_tiny_224', num_classes=1000, include_top=False, pretrained=pretrained)

            if model_type == 2:
                #if weights=='imagenet':
                   #weights=os.path.join(os.getcwd(), 'models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5')
                model=tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)

            if model_type == 3:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5')
                model=tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 4:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/efficientnetv2-s_notop.h5')
                model=tf.keras.applications.EfficientNetV2S(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 5:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/evgg19_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.VGG19(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 6:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/xception_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.Xception(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 7:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 8:
                model=MobileVit(input_shape=input_shape, include_top=False,classifier_activation=None)

            if model_type == 9:
                model = ViT(input_shape=input_shape, include_top=False, classifier_activation=None)

            if model_type == 10:
                model=MobileVitC(input_shape=input_shape, include_top=False,classifier_activation=None)


            single_channel_header = tf.keras.Sequential()
            single_channel_header.add(Flatten())
            single_channel_header.add(Dense(out_shape, activation='sigmoid'))

            single_out = single_channel_header(model(inp))

            super(BackboneModel, self).__init__(inputs=inp, outputs=single_out)


        def compute_output_shape(self, input_shape):
            inp=tf.keras.layers.Input(shape=input_shape[1:])
            out=tf.keras.layers.Input(shape=4)
            #out=self.call(inp)
            return out.shape[:]




"""
Created on Feb 03, 2021 
@file: DANet_attention3D.py
@desc: Dual attention network.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import tensorflow as tf


class Channel_attention(tf.keras.layers.Layer):
    """
    Channel attention module

    Fu, Jun, et al. "Dual attention network for scene segmentation."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        super(Channel_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(Channel_attention, self).get_config().copy()
        config.update({
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint
        })
        return config

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        super(Channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs


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
            grad=1,
            **kwargs):

        super(ECA, self).__init__()

        self.kwargs = kwargs
        self.b = b
        self.gamma=gamma
        self.kernel=kernel
        self.grad=grad

    def build(self, input_shapes):

        if self.kernel==None:
            t = int(abs(math.log(input_shapes[-1], 2) + self.b)/self.gamma)
            k = t if t % 2 else t + 1
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

    def gradient(self,x):
        fd = tf.concat([x[:, 1:, :], tf.expand_dims(x[:, -1, :], -1)], -2)
        bd = tf.concat([tf.expand_dims(x[:, 0, :], -1), x[:, :-1:, :]], -2)
        d = tf.concat([fd, bd], -1)
        d = tf.reduce_mean(d, -1,keepdims=True)

        return d

    def call(self, x):

        # (B, C, 1)
        attn = tf.math.reduce_mean(x, [-3, -2])[:, :,tf.newaxis]
        for i in range(self.grad):
            attn=self.gradient(attn)

        # (B, C, 1)
        attn = self.eca_conv(attn)
        # (B, 1, 1, C)
        attn=tf.transpose(attn, [0, 2, 1])
        attn = tf.expand_dims(attn, -3)

        # (B, 1, 1, C)
        attn = tf.math.sigmoid(attn)

        return x * attn


class FNetEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super(FNetEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = tf.keras.Sequential(
            [
                Dense(dense_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()

    def call(self, inputs):
        # (B, C, 1)
        #inputs = tf.math.reduce_mean(x, [-3, -2])[:,tf.newaxis, : ]
        # Casting the inputs to complex64
        inp_complex = tf.cast(inputs, tf.complex64)
        # Projecting the inputs to the frequency domain using FFT2D and
        # extracting the real part of the output
        fft = tf.math.real(tf.signal.fft3d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = tf.math.reduce_mean(proj_input, [-3, -2])
        proj_output = self.dense_proj(proj_output)
        proj_output=proj_output[:,tf.newaxis,tf.newaxis, : ]
        fout=self.layernorm_2(proj_input + proj_output)
        return fout


class ChannelGate(tf.keras.layers.Layer):
    """Apply Channelwise attention to input.
    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C, H, W)
    Attributes:
        gate_channels (int): number of channels of input
        reduction_ratio (int): factor to reduce the channels in FF layer
        pool_types (list): list of pooling operations
    """

    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=['avg', 'max'],
        **kwargs):

        super(ChannelGate, self).__init__()

        all_pool_types = {'avg', 'max'}
        if not set(pool_types).issubset(all_pool_types):
            raise ValueError('The available pool types are: {}'.format(all_pool_types))

        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types
        self.kwargs = kwargs

    def build(self, input_shape):
        hidden_units = self.gate_channels // self.reduction_ratio
        self.mlp = models.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(self.gate_channels, activation=None)
        ])


    def apply_pooling(self, inputs, pool_type):
        """Apply pooling then feed into ff.
        Args:
            inputs (tf.ten
        Returns:
            (tf.tensor) shape (B, C)
        """

        if pool_type == 'avg':
            pool = tf.math.reduce_mean(inputs, [2, 3])
        elif pool_type == 'max':
            pool = tf.math.reduce_max(inputs, [2, 3])

        channel_att = self.mlp(pool)
        return channel_att

    def call(self, inputs):
        pools = [self.apply_pooling(inputs, pool_type) \
            for pool_type in self.pool_types]

        # (B, C, 1, 1)
        attn = tf.math.sigmoid(tf.math.add_n(pools))[:, :, tf.newaxis, tf.newaxis]

        return attn * inputs





