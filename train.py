
import keras.backend as K

from keras.optimizers import Adam

from nets.model import get_model
from nets.model_training import ProposalTargetCreator

from utils.dataloader import FRCNNDatasets
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch




if __name__ == "__main__":
    #para
    classes_path    = 'model_data/classes.txt'
    model_path      = 'model_data/model.h5'
    input_shape     = [512, 512]

    batch_size = 2
    lr = 1e-3
    start_epoch = 0
    end_epoch = 50

    train_annotation_path   = 'train.txt'
    val_annotation_path     = 'val.txt'



    #model
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1

    model_rpn, model_all = get_model(num_classes)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_rpn.load_weights(model_path, by_name=True)
        model_all.load_weights(model_path, by_name=True, skip_mismatch= True)

    roi_helper      = ProposalTargetCreator(num_classes)



    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    model_rpn.compile(optimizer=Adam(lr=lr))
    model_all.compile(optimizer=Adam(lr), loss={'total_loss': lambda y_true, y_pred: y_pred})

    gen = FRCNNDatasets(train_lines, input_shape,  batch_size, num_classes, train=True).generate()
    gen_val = FRCNNDatasets(val_lines, input_shape,  batch_size, num_classes, train=False).generate()

    #train
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_rpn, model_all, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      end_epoch,
                      roi_helper)

        if epoch > 35:
            lr = 5e-5
        elif epoch > 15:
            lr = 5e-4
        K.set_value(model_rpn.optimizer.lr, lr)
        K.set_value(model_all.optimizer.lr, lr)

        model_all.save_weights('ep%03d.h5' % (epoch + 1))



