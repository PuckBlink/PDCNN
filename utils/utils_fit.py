import numpy as np
from keras import backend as K
from tqdm import tqdm



def fit_one_epoch(model_rpn, model_all, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, roi_helper):
    total_loss = 0
    hm_loss = 0
    wh_loss = 0
    reg_loss = 0
    outclsloss= 0
    outregloss = 0
    lossdata = []

    val_loss = 0
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            X, hm_imput, wh_input, reg_input, reg_mask_input, index_input, boxes = batch[0], batch[1], batch[2], batch[3], batch[4], batch[
                5], batch[6]


            rpn_pred = model_rpn.predict_on_batch(X)
            trainrpn = False
            random_augment = True
            if not trainrpn:
                poscount = 36
                rpn_results = rpn_pred[:, :poscount, 1:]


                if random_augment:
                    dx1 = np.random.random((rpn_results.shape[1])) * 0.1 - 0.025
                    dx2 = np.random.random((rpn_results.shape[1])) * 0.1 - 0.025
                    dy1 = np.random.random((rpn_results.shape[1])) * 0.1 - 0.025
                    dy2 = np.random.random((rpn_results.shape[1])) * 0.1 - 0.025

                    width = (rpn_results[:, :, 2] - rpn_results[:, :, 0])
                    hight = (rpn_results[:, :, 3] - rpn_results[:, :, 1])

                    rpn_results[:, :, 0] = rpn_results[:, :, 0] - dx1 * width
                    rpn_results[:, :, 2] = rpn_results[:, :, 2] + dx2 * width

                    rpn_results[:, :, 1] = rpn_results[:, :, 1] - dy1 * hight
                    rpn_results[:, :, 3] = rpn_results[:, :, 3] + dy2 * hight

                roi_inputs = []
                out_classes = []
                out_regrs = []

                lesspos = rpn_results.shape[1]

                for i in range(len(X)):
                    R = rpn_results[i]
                    X2, Y1, Y2 = roi_helper.calc_iou(R, boxes[i])
                    if X2.shape[0] < lesspos:
                        lesspos = X2.shape[0]

                    roi_inputs.append(X2)
                    out_classes.append(Y1)
                    out_regrs.append(Y2)

                for i in range(len(X)):
                    roi_inputs[i] = roi_inputs[i][:lesspos, :]
                    out_classes[i] = out_classes[i][:lesspos, :]
                    out_regrs[i] = out_regrs[i][:lesspos, :]
            else:
                roi_inputs = np.zeros((2, 1, 4))
                out_classes = np.zeros((2, 1, 5))
                out_regrs = np.zeros((2, 1, 32))

            y_true = [np.zeros(2) for _ in range(5)]

            loss_class = model_all.train_on_batch(x=[X, np.array(roi_inputs), hm_imput, wh_input, reg_input,reg_mask_input,index_input,
                                                     np.array(out_classes), np.array(out_regrs)],
                                                  y = y_true)
            hm_loss += loss_class[1]
            wh_loss += loss_class[2]
            reg_loss += loss_class[3]
            outclsloss += loss_class[4]
            outregloss += loss_class[5]

            total_loss += loss_class[0]

            pbar.set_postfix(**{'hm_loss'    : hm_loss  / (iteration + 1),
                                'wh_loss'  : wh_loss  / (iteration + 1),
                                'reg_loss'  : reg_loss  / (iteration + 1),
                                'outclsloss'  : outclsloss  / (iteration + 1),
                                'outregloss'  : outregloss / (iteration + 1) ,
                                'lr'       : K.get_value(model_all.optimizer.lr)})
            pbar.update(1)
    lossdata.append([epoch+1, total_loss/ epoch_step, hm_loss  / (iteration + 1), wh_loss  / (iteration + 1),
                     reg_loss  / (iteration + 1), outclsloss  / (iteration + 1)])

    hm_loss = 0
    wh_loss = 0
    reg_loss = 0
    outclsloss= 0
    outregloss = 0

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break


            X, y1, y2, y3, reg_mask_input, index_input, boxes = batch[0], batch[1], batch[2], batch[3], batch[4], batch[
                5], batch[6]
            rpn_pred = model_rpn.predict_on_batch(X)
            
            if not trainrpn:

                poscount = 36
                for j in range(rpn_pred.shape[0]):
                    for i in range(rpn_pred.shape[1]):
                        temp = rpn_pred[j, -1 - i, 0]
                        if temp > 0.4:
                            tempposcount = 36 - i
                            break
                    if tempposcount < poscount:
                        poscount = tempposcount
                if poscount == 0: poscount = 1

                rpn_results = rpn_pred[:, :poscount, 1:]

                roi_inputs = []
                out_classes = []
                out_regrs = []
                lesspos = rpn_results.shape[1]

                for i in range(len(X)):
                    R = rpn_results[i]
                    X2, Y1, Y2 = roi_helper.calc_iou(R, boxes[i])
                    if X2.shape[0] < lesspos:
                        lesspos = X2.shape[0]

                    roi_inputs.append(X2)
                    out_classes.append(Y1)
                    out_regrs.append(Y2)

                for i in range(len(X)):
                    roi_inputs[i] = roi_inputs[i][:lesspos, :]
                    out_classes[i] = out_classes[i][:lesspos, :]
                    out_regrs[i] = out_regrs[i][:lesspos, :]

            loss_class = model_all.test_on_batch(x=[X, np.array(roi_inputs), y1,y2,y3,reg_mask_input,index_input,
                                                     np.array(out_classes), np.array(out_regrs)],
                                                  y = y_true)

            hm_loss += loss_class[1]
            wh_loss += loss_class[2]
            reg_loss += loss_class[3]
            outclsloss += loss_class[4]
            outregloss += loss_class[5]
            val_loss += loss_class[0]
            pbar.set_postfix(**{'hm_loss'    : hm_loss  / (iteration + 1),
                                'wh_loss'  : wh_loss  / (iteration + 1),
                                'reg_loss'  : reg_loss  / (iteration + 1),
                                'outclsloss'  : outclsloss  / (iteration + 1),
                                'outregloss'  : outregloss / (iteration + 1) ,
                                'lr'       : K.get_value(model_all.optimizer.lr)})
            pbar.update(1)

    logs = {'loss': total_loss / epoch_step, 'val_loss': val_loss / epoch_step_val}

    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))





