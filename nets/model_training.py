import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

def totalloss(args):

    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask,\
    indices, outclspre, outregpre, outcls_input, outreg_input = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)

    outclsloss = 5*classifier_cls_loss(outcls_input, outclspre)
    outregloss = 0.5*class_loss_regr_fixed_num(outreg_input, outregpre)

    return [hm_loss, wh_loss, reg_loss, outclsloss, outregloss]


def focal_loss(hm_pred, hm_true):

    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-6, 1.)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1 - hm_pred, 1e-6, 1.)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)
    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    b, c = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
    k = tf.shape(indices)[1]

    y_pred = tf.reshape(y_pred, (b, -1, c))
    length = tf.shape(y_pred)[1]
    indices = tf.cast(indices, tf.int32)

    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, k))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(length) +
                    tf.reshape(indices, [-1]))

    y_pred = tf.gather(tf.reshape(y_pred, [-1, c]), full_indices)
    y_pred = tf.reshape(y_pred, [b, -1, c])

    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

def classifier_cls_loss(y_true, y_pred):
    unweighted_losses = K.categorical_crossentropy(y_true, y_pred)
    class_weights = tf.constant([[[0.36, 0.39, 0.85, 2.4, 1]]])

    mul = class_weights * y_true
    weights = tf.reduce_sum(mul, axis=-1)
    weighted_losses = unweighted_losses * weights
    return K.mean(weighted_losses)

def class_loss_regr_fixed_num(y_true, y_pred, num_classes = 4, sigma_squared=1,epsilon = 1e-4):
    regression = y_pred
    regression_target = y_true[:, :, 4 * num_classes:]

    regression_diff = regression_target - regression
    regression_diff = keras.backend.abs(regression_diff)

    regression_loss = 4 * K.sum(y_true[:, :, :4 * num_classes] * tf.where(
        keras.backend.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )
                                )
    normalizer = K.sum(epsilon + y_true[:, :, :4 * num_classes])
    regression_loss = keras.backend.sum(regression_loss) / normalizer

    return regression_loss




class ProposalTargetCreator(object):
    def __init__(self, num_classes, n_sample=36, pos_ratio=0.7, pos_iou_thresh=0.4,
        neg_iou_thresh_high=0.4, neg_iou_thresh_low=0, variance=[0.125, 0.125, 0.25, 0.25]):

        self.n_sample               = n_sample
        self.pos_ratio              = pos_ratio
        self.pos_roi_per_image      = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh         = pos_iou_thresh
        self.neg_iou_thresh_high    = neg_iou_thresh_high
        self.neg_iou_thresh_low     = neg_iou_thresh_low
        self.num_classes            = num_classes
        self.variance               = variance

    def bbox_iou(self, bbox_a, bbox_b):
        if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
            print(bbox_a, bbox_b)
            raise IndexError
        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
        area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
        return area_i / (area_a[:, None] + area_b - area_i)

    def bbox2loc(self, src_bbox, dst_bbox):
        width = src_bbox[:, 2] - src_bbox[:, 0]
        height = src_bbox[:, 3] - src_bbox[:, 1]
        ctr_x = src_bbox[:, 0] + 0.5 * width
        ctr_y = src_bbox[:, 1] + 0.5 * height

        base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
        base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
        base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
        base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

        eps = np.finfo(height.dtype).eps
        width = np.maximum(width, eps)
        height = np.maximum(height, eps)

        dx = (base_ctr_x - ctr_x) / width
        dy = (base_ctr_y - ctr_y) / height
        dw = np.log(base_width / width)
        dh = np.log(base_height / height)

        loc = np.vstack((dx, dy, dw, dh)).transpose()
        return loc

    def calc_iou(self, R, all_boxes):
        Negtiveuse = False

        bboxes  = all_boxes[:, :4]
        label   = all_boxes[:, 4]

        if len(bboxes)==0:
            max_iou         = np.zeros(len(R))
            gt_assignment   = np.zeros(len(R), np.int32)
            gt_roi_label    = np.zeros(len(R))
        else:
            iou             = self.bbox_iou(R, bboxes)
            max_iou         = iou.max(axis=1)
            gt_assignment   = iou.argmax(axis=1)
            gt_roi_label    = label[gt_assignment]

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]

        pos_roi_per_this_image = int(pos_index.size)


        if not Negtiveuse:
            keep_index = pos_index
        else:
            keep_index = np.where(max_iou >= -1)[0]

        if pos_index.shape[0]<keep_index.shape[0]:
            gt_roi_label[pos_index.shape[0]:] = 4

        sample_roi = R[keep_index]

        if len(bboxes) != 0:
            gt_roi_loc = self.bbox2loc(sample_roi, bboxes[gt_assignment[keep_index]])
            gt_roi_loc = gt_roi_loc / np.array(self.variance)
        else:
            gt_roi_loc = np.zeros_like(sample_roi)

        gt_roi_label                            = gt_roi_label[keep_index]

        X                   = np.zeros_like(sample_roi)
        X[:, [0, 1, 2, 3]]  = sample_roi[:, [1, 0, 3, 2]]

        Y1                  = np.eye(self.num_classes)[np.array(gt_roi_label, np.int32)]


        y_class_regr_label  = np.zeros([np.shape(gt_roi_loc)[0], self.num_classes-1, 4])
        y_class_regr_coords = np.zeros([np.shape(gt_roi_loc)[0], self.num_classes-1, 4])
        y_class_regr_label[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image],
                           np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = 1
        y_class_regr_coords[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image],
                            np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = \
            gt_roi_loc[:pos_roi_per_this_image]
        try:
            y_class_regr_label = np.reshape(y_class_regr_label, [np.shape(gt_roi_loc)[0], -1])
            y_class_regr_coords = np.reshape(y_class_regr_coords, [np.shape(gt_roi_loc)[0], -1])
        except:
            return np.zeros((1, 4)), np.zeros((1, 5)), np.zeros((1, 32))

        Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis = 1)
        return X, Y1, Y2
