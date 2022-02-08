
from keras.models import Model
from keras.layers import Input, Lambda

from nets.classifier import get_resnet50_classifier
from nets.resnet import Backbone
from nets.rpn import decode, outhead

from nets.model_training import totalloss


USEFM =True
ROISIZE = 32

def get_model(num_classes):
    inputs      = Input(shape=(512, 512, 3))
    roi_input   = Input(shape=(None, 4))
    reg_mask_input = Input(shape=(33,))
    index_input = Input(shape=(33,))
    hm_input        = Input(shape=(128, 128, 1))
    wh_input        = Input(shape=(33, 2))
    reg_input       = Input(shape=(33, 2))
    outcls_input = Input(shape=(None, 5))
    outreg_input = Input(shape=(None, 4*8))

    if True:
        base_layers,y = Backbone(inputs)
        y1, y2, y3 = outhead(base_layers)
        rpn = Lambda(lambda x: decode(*x, max_objects=36), name = 'rpn')([y1, y2, y3])
        if not USEFM:
            y = inputs
        classifier  = get_resnet50_classifier(y, roi_input, ROISIZE, num_classes)
    model_rpn   = Model(inputs, rpn)
    loss_ = Lambda(totalloss, name='total_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input, classifier[0],classifier[1], outcls_input, outreg_input])
    model_all   = Model(inputs = [inputs, roi_input, hm_input, wh_input, reg_input, reg_mask_input,
                                  index_input, outcls_input, outreg_input], outputs =loss_)
    return model_rpn, model_all

def get_predict_model(num_classes):
    inputs              = Input(shape=(None, None, 3))
    roi_input           = Input(shape=(None, 4))
    
    if True:
        yinput = Input(shape=(None, None, None))
        base_layers, y = Backbone(inputs)
        y1, y2, y3 = outhead(base_layers)
        rpn = Lambda(lambda x: decode(*x, max_objects=36), name='rpn')([y1, y2, y3])
        if not USEFM:
            y=inputs
        classifier  = get_resnet50_classifier(y, roi_input, ROISIZE, num_classes)

    model_rpn   = Model(inputs, outputs =[rpn, y])


    model_classifier_only = Model(inputs =[inputs, yinput, roi_input], outputs =[classifier[0],classifier[1]])

    return model_rpn, model_classifier_only
