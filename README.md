# ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
This repository contains the TensorFlow implementation of paper [ESPNet](https://arxiv.org/abs/1803.06815).

# Compatibility
The code is tested using Tensorflow v1.8 with Python v2.7.

# Dataset
[ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#phase/5841916ccad3a51cc66c8db0)

# Structure of this repository
* [dataset.py](/dataset.py/) Source code for preparing dataset. If you are not using the ISIC 2017 skin segmentation dataset, kindly make required changes in the dataset.py file. 
* [train.py](/train.py/) Source code for training  ESPNet-C and ESPNet model.
* [model.py](/model.py/) Source code for model architecture of ESPNet-C and ESPNet model.
* [config.py](/config.py/) Source code for configuration file containing image size required for training.
* [test.py](/test.py/) Source code for creating segmentation masks of images for testing.
* [eval.py](/eval.py/) Source code for evaluating model.


# Usage
The network can be trained using the train.py script. 
```
python train.py --model_file_name <model file name> \
--epochs <number of epochs> \
--batch_size <batch size> \
--model_name <espnet_c/espnet>\
--image_dir <Folder containing training images>\
--ann_dir <Folder containing annotations of training images>
``` 

Segmented mask can be generated using test.py script.
```
python test.py --model_file_name <model file name> \
--batch_size <batch size> \
--input_folder <Folder containing images to be tested> \
--op_folder <Output Folder> 
```
Model can be evaluated using eval.py script.
```
python eval.py --ground_truth <ground truth directory> \
--prediction <prediction image directory containing images of mask predicted by the model>
```

