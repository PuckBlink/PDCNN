from keras import backend as K
from keras.engine import Layer
from keras.initializers import random_normal
from keras.layers import Activation, BatchNormalization, Conv2D
import keras


def Backbone(inputs):
    bn = 1
    gn = 0
    af = 0

    x = Conv2D(32, (3, 3), strides=(2, 2), padding= 'same', kernel_initializer=random_normal(stddev=0.02), name='conv1', use_bias=False)(inputs)
    x = BatchNormalization(name='bn_conv1')(x)
    s2 = Activation('relu')(x)


    x = Conv2dUnit(32, 64, 3, stride=2, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage1_conv1')(s2)
    s4 = Conv2dUnit(64, 64, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage1_conv2')(x)
    x = Conv2dUnit(64, 64, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage1_conv3')(x)
    x = StackResidualBlock(64, 32, 64, n=2, bn=bn, gn=gn, af=af, name='backbone.stage1_blocks')(x)
    x = Conv2dUnit(64, 64, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage1_conv4')(x)
    x = keras.layers.Concatenate(axis=-1)([x, s4])
    s4 = Conv2dUnit(128, 64, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage1_conv5')(x)


    x = Conv2dUnit(64, 128, 3, stride=2, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage2_conv1')(s4)
    s8 = Conv2dUnit(128, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage2_conv2')(x)
    x = Conv2dUnit(128, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage2_conv3')(x)
    x = StackResidualBlock(128, 64, 128, n=2, bn=bn, gn=gn, af=af, name='backbone.stage2_blocks')(x)
    x = Conv2dUnit(128, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage2_conv4')(x)
    x = keras.layers.Concatenate(axis=-1)([x, s8])
    s8 = Conv2dUnit(256, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage2_conv5')(x)


    x = Conv2dUnit(128, 256, 3, stride=2, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage3_conv1')(s8)
    s16 = Conv2dUnit(256, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage3_conv2')(x)
    x = Conv2dUnit(256, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage3_conv3')(x)
    x = StackResidualBlock(256, 128, 256, n=2, bn=bn, gn=gn, af=af, name='backbone.stage3_blocks')(x)
    x = Conv2dUnit(256, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage3_conv4')(x)
    x = keras.layers.Concatenate(axis=-1)([x, s16])
    s16 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                                       name='backbone.stage3_conv5')(x)


    y = s8

    x = Conv2dUnit(256, 512, 3, stride=2, bn=bn, gn=gn, af=af, act='mish',
                   name='backbone.stage4_conv1')(s16)
    s32 = Conv2dUnit(512, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                     name='backbone.stage4_conv2')(x)
    x = Conv2dUnit(512, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                   name='backbone.stage4_conv3')(x)
    x = StackResidualBlock(512, 256, 512, n=2, bn=bn, gn=gn, af=af, name='backbone.stage4_blocks')(x)
    x = Conv2dUnit(512, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                   name='backbone.stage4_conv4')(x)
    x = keras.layers.Concatenate(axis=-1)([x, s32])
    s32 = Conv2dUnit(1024, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='mish',
                     name='backbone.stage4_conv5')(x)

    s32 = Conv2dUnit(512, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv078')(s32)
    s16 = Conv2dUnit(256, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv079')(s16)
    s32 = keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(s32)
    s16 = keras.layers.Concatenate(axis=-1)([s16,s32])


    s16 = Conv2dUnit(768, 768, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv080')(s16)
    s8 = Conv2dUnit(128, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv081')(s8)
    s16 = keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(s16)
    s8 = keras.layers.Concatenate(axis=-1)([s8,s16])

    s8 = Conv2dUnit(896, 896, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv082')(s8)
    s4 = Conv2dUnit(64, 64, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv083')(s4)
    s8 = keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(s8)
    s4 = keras.layers.Concatenate(axis=-1)([s4,s8])

    s4 = Conv2dUnit(960, 64, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv084')(s4)

    return s4,y



class ResidualBlock(object):
    def __init__(self, input_dim, filters_1, filters_2, bn, gn, af, name=''):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dUnit(input_dim, filters_1, 1, stride=1, bias_attr=False, bn=bn, gn=gn, af=af, act='mish', name=name+'.conv1')
        self.conv2 = Conv2dUnit(filters_1, filters_2, 3, stride=1, bias_attr=False, bn=bn, gn=gn, af=af, act='mish', name=name+'.conv2')

    def __call__(self, input):
        residual = input
        x = self.conv1(input)
        x = self.conv2(x)
        x = keras.layers.add([residual, x])
        return x

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()


class StackResidualBlock(object):
    def __init__(self, input_dim, filters_1, filters_2, n, bn, gn, af, name=''):
        super(StackResidualBlock, self).__init__()
        self.sequential = []
        for i in range(n):
            residual_block = ResidualBlock(input_dim, filters_1, filters_2, bn, gn, af, name=name+'.block%d' % (i,))
            self.sequential.append(residual_block)

    def __call__(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x

    def freeze(self):
        for residual_block in self.sequential:
            residual_block.freeze()

class Conv2dUnit(object):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 bn=0,
                 gn=0,
                 af=0,
                 groups=32,
                 act=None,
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 use_dcn=False,
                 name=''):
        super(Conv2dUnit, self).__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.groups = groups
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn
        pad = (filter_size - 1) // 2
        self.padding = pad
        padding = None
        self.zero_padding = None
        if pad == 0:
            padding = 'valid'
        else:
            if stride == 1:
                padding = 'same'
            elif stride == 2:
                if not use_dcn:
                    padding = 'valid'
                    self.zero_padding = keras.layers.ZeroPadding2D(padding=((pad, 0), (pad, 0)))
        kernel_initializer = 'glorot_uniform'
        bias_initializer = 'zeros'
        if use_dcn:
            self.conv = keras.layers.SeparableConv2D(filters, kernel_size=filter_size, strides=stride, padding=padding,
                                            use_bias=bias_attr,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                            name=name + '.conv')
        else:
            self.conv = keras.layers.Conv2D(filters, kernel_size=filter_size, strides=stride, padding=padding, use_bias=bias_attr,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name=name+'.conv')

        if bn:

            self.bn = keras.layers.BatchNormalization(name=name+'.bn', epsilon=1e-5)
        if gn:
            pass

        if af:
            pass

        self.act = None
        if act == 'relu':
            self.act = keras.layers.ReLU()
        elif act == 'leaky':
            self.act = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)

        elif act == 'mish':
            self.act = Mish()

    def freeze(self):
        self.conv.trainable = False
        if self.bn is not None:
            self.bn.trainable = False

    def __call__(self, x):
        if self.zero_padding:
            x = self.zero_padding(x)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.af:
            x = self.af(x)
        if self.act:
            x = self.act(x)
        return x

class Mish(Layer):
    def __init__(self):
        super(Mish, self).__init__()
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, x):
        return x * (K.tanh(K.softplus(x)))









