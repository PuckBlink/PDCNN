from keras.initializers import random_normal
import tensorflow as tf

from keras.layers import Activation, BatchNormalization, Conv2D,MaxPooling2D

from keras.regularizers import l2


def outhead(x):

    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(1, 1, kernel_initializer=random_normal(stddev=0.02), kernel_regularizer=l2(5e-4), activation='sigmoid', name='hmheader')(y1)


    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(2, 1, kernel_initializer=random_normal(stddev=0.02), kernel_regularizer=l2(5e-4), name='whheader')(y2)


    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer=random_normal(stddev=0.02), kernel_regularizer=l2(5e-4), name='regheader')(y3)
    return y1, y2, y3


def nms(heat, kernelh=14, kernelw = 3):
    hmax = MaxPooling2D((kernelh, kernelw), strides=1, padding='SAME')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects=36):

    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.to_int32(length) + tf.reshape(indices, [-1])
    topk_reg = tf.gather(tf.reshape(reg, [-1, 2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])
    topk_wh = tf.gather(tf.reshape(wh, [-1, 2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2

    topk_x1 = topk_x1/128
    topk_x2 = topk_x2/128
    topk_y1 = topk_y1/128
    topk_y2 = topk_y2/128
    scores = tf.expand_dims(scores, axis=-1)
    detections = tf.concat([scores, topk_x1, topk_y1, topk_x2, topk_y2], axis=-1)
    return [detections]