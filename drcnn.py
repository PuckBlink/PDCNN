import colorsys
import os

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

import nets.model as model

from utils.utils import cvtColor, get_classes, resize_image
from utils.utils_bbox import BBoxUtility

import cv2








class DRCNN(object):
    _defaults = {
        "model_path"    : 'model_data/model.h5',
        "classes_path"  : 'model_data/classes.txt',
        "backbone"      : "resnet50",
        "confidence"    : 0.3,
        "nms_iou"       : 0.6,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"




    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.num_classes                    = self.num_classes + 1

        self.bbox_util = BBoxUtility(self.num_classes, nms_iou = self.nms_iou, min_k = 150)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model_rpn, self.model_classifier = model.get_predict_model(self.num_classes, self.backbone)
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))

    def getpredict(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = [512, 512]
        image = cvtColor(image)
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        [rpn_pred, y_input] = self.model_rpn.predict(image_data)

        poscount = 0
        for i in range(rpn_pred.shape[1]):
            temp = rpn_pred[0, -1 - i, 0]
            if temp > self.confidence:
                poscount = 36 - i
                break
        rpn_results = rpn_pred[:, :poscount, 1:]
        classifier_pred = self.model_classifier.predict([image_data, y_input, rpn_results[:, :, [1, 0, 3, 2]]])

        results = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape,
                                                          self.confidence)

        if len(results[0]) == 0:
            return image

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        return top_label, top_conf, top_boxes
    



    def detect_image(self, image):

        top_label, top_conf, top_boxes = self.getpredict(image)

        image = drawpre(image,top_boxes,top_conf,top_label)

        return image




def drawpre(image, boxes, scores, classes):
    image = np.array(image)
    num_classes = 4
    all_classes = ['I','II','III','IV']

    image_h, image_w, _ = image.shape

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for box, score, cl in zip(boxes, scores, classes):
        y0, x0, y1, x1 = box
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
        bbox_color = colors[cl]

        bbox_thick = 2
        cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
        bbox_mess = '%s:%.2f' % (all_classes[cl], score)


        cv2.putText(image, bbox_mess, (left, top - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color=bbox_color, thickness=2, lineType=cv2.LINE_AA)
    image = Image.fromarray(image)
    return image