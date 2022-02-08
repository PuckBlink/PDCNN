import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.initializers import random_normal

from keras.layers import (Dense, Flatten, Activation, BatchNormalization, Conv2D,
                          Add, AveragePooling2D, TimeDistributed)


class RoiPoolingConv(Layer):
    def __init__(self, pool_size, **kwargs):

        self.pool_size = pool_size

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        input_shape2 = input_shape[1]
        return None, input_shape2[1], self.pool_size, self.pool_size, self.nb_channels
    def call(self, x, mask=None):
        assert(len(x) == 2)
        feature_map = x[0]
        rois        = x[1]
        num_rois    = tf.shape(rois)[1]
        batch_size  = tf.shape(rois)[0]
        box_index   = tf.expand_dims(tf.range(0, batch_size), 1)
        box_index   = tf.tile(box_index, (1, num_rois))
        box_index   = tf.reshape(box_index, [-1])
        rs          = tf.image.crop_and_resize(feature_map, tf.reshape(rois, [-1, 4]), box_index, (self.pool_size, self.pool_size))
        final_output = K.reshape(rs, (batch_size, num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output

def get_resnet50_classifier(base_layers, input_rois, roi_size, num_classes):
    out_roi_pool = RoiPoolingConv(roi_size)([base_layers, input_rois])
    out = resnet32_classifier_layers(out_roi_pool)
    out = TimeDistributed(Flatten())(out)
    out_class   = TimeDistributed(Dense(5, activation='softmax', kernel_initializer=random_normal(stddev=0.02)), name='dense_class_{}'.format(num_classes))(out)
    out_regr    = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=random_normal(stddev=0.02)), name='dense_regress_{}'.format(num_classes))(out)
    return [out_class, out_regr]


def resnet32_classifier_layers(x):
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer='normal'),
                        name='')(x)
    x = identity_block_td(x, 3, [32, 32, 64], stage=1, block='b')
    x = identity_block_td(x, 3, [32, 32, 64], stage=1, block='c')
    x = identity_block_td(x, 3, [32, 32, 64], stage=1, block='d')
    x = conv_block_td(x, 3, [64, 64, 128], stage=2, block='a', strides=(2, 2))

    x = identity_block_td(x, 3, [64, 64, 128], stage=2, block='b')
    x = identity_block_td(x, 3, [64, 64, 128], stage=2, block='c')
    x = identity_block_td(x, 3, [64, 64, 128], stage=2, block='d')

    x = conv_block_td(x, 3, [128, 128, 256], stage=3, block='a', strides=(2, 2))

    x = identity_block_td(x, 3, [128, 128, 256], stage=3, block='b')
    x = identity_block_td(x, 3, [128, 128, 256], stage=3, block='c')
    x = identity_block_td(x, 3, [128, 128, 256], stage=3, block='d')

    x = conv_block_td(x, 3, [256, 256, 512], stage=4, block='a', strides=(2, 2))

    x = identity_block_td(x, 3, [256, 256, 512], stage=4, block='b')
    x = identity_block_td(x, 3, [256, 256, 512], stage=4, block='c')
    x = identity_block_td(x, 3, [256, 256, 512], stage=4, block='d')



    x = TimeDistributed(AveragePooling2D((2, 2)), name='avg_pool')(x)

    return x



def identity_block_td(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base  = 'res' + str(stage) + block + '_branch'
    bn_name_base    = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base  = 'res' + str(stage) + block + '_branch'
    bn_name_base    = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x