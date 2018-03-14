# -*- coding:utf-8 -*-
"""
Util module.
"""
import tensorflow as tf
import numpy as np
# import scipy.misc
import cv2

def save_images(input, output1, output2, input_path, image_path, max_samples=4):
    image = np.concatenate([output1, output2], axis=2) # concat 4D array, along width.
    if max_samples > int(image.shape[0]):
    	max_samples = int(image.shape[0])
    
    image = image[0:max_samples, :, :, :]
    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)
    # concat 3D array, along axis=0, i.e. along height. shape: (1024, 256, 3).

    # save image.
    # scipy.misc.toimage(), array is 2D(gray, reshape to (H, W)) or 3D(RGB).
    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(image_path) # image_path contain image path and name.
    cv2.imwrite(image_path, np.uint8(image.clip(0., 1.) * 255.))

    # save input
    if input is not None:
        input_data = input[0:max_samples, :, :, :]
        input_data = np.concatenate([input_data[i, :, :, :] for i in range(max_samples)], axis=0)
        cv2.imwrite(input_path, np.uint8(input_data.clip(0., 1.) * 255.))