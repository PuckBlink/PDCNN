# Automatic PBL Diagnosis
Automated Periodontitis Bone Loss Diagnosis in Panoramic Radiographs Using a Bespoke Two-Stage Detector

## Introduction

This network is use for automatic periodontitis bone loss diagnosis,
if more background information is needed please refer to the paper.  
    
This code is written in python language with Keras framework. This code contains the network training part and the prediction part, 
and the experiment model weights and datasets can be provided if needed.


## Implementations
**1. Preparation of dataset**  

train.txt and val.txt contain the path and annotation information of 
the image train and validation samples.      
Keep your images in a fold, and create two txt documents
to write information in the following form.

```
xxx/Images/1.jpg 810,620,957,901,0 944,617,1106,916,1 ...
xxx/Images/2.jpg 810,620,957,901,1 944,617,1106,916,0 ...
...
```

**2. Train**  

After preparing the dataset, complete the python environment, 
path information, hyperparameters and other settings, 
you can start training with train.py. 
You can save intermediate models on demand.

**3. Predict** 

You can generate a display of the model's predictive 
effect on the test sample with predict.py and 
to ensure the script runs correctly please check the path dependencies.




## Reference:

https://github.com/qqwweee/keras-yolo3/  
https://github.com/pierluigiferrari/ssd_keras  
https://github.com/jinfagang/keras_frcnn  
https://github.com/Cartucho/mAP  
https://github.com/bubbliiiing/centernet-keras
