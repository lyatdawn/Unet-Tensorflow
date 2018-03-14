# Unet 
* Tensorflow implement of U-Net: Convolutional Networks for Biomedical Image Segmentation..[[Paper]](https://arxiv.org/abs/1505.04597)
* Borrowed code and ideas from zhixuhao's unet: https://github.com/zhixuhao/unet.

## Install Required Packages
First ensure that you have installed the following required packages:
* TensorFlow1.4.0 ([instructions](https://www.tensorflow.org/install/)). Maybe other version is ok.
* Opencv ([instructions](https://github.com/opencv/opencv)). Here is opencv-2.4.9.

## Datasets
* In this implementation of the Unet, we use Carvana Image Masking Challenge data.[[download]](https://www.kaggle.com/c/carvana-image-masking-challenge/data) We download train.zip and train_masks.zip.
* Run **scripts/transform_images.py** to transform all the image to gray JPEG image.
* Run **scripts/build_tfrecords.py** to generate training data, data format is tfrecords.

## Training and Testing Model
* Run the following script to train the model, in the process of training, will save the training images every 500 steps. See the **model/unet.py** for details.
```shell
sh train.sh
```
You can change the arguments in train.sh depend on your machine config.
* Run the following script to test the trained model. The test.sh will transform the datasets.
```shell
sh test.sh
```
The script will load the trained StarGAN model to generate the transformed images. You could change the arguments in test.sh depend on your machine config.

## Downloading data/trained model
* Pretrained model: [[download]](https://drive.google.com/open?id=1ngSzJN3oUdn2Xrrvl_vNyPsQThI0hHcY). The model-8500 is better.