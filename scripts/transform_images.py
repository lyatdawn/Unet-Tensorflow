# -*- coding:utf-8 -*-
import os
# import cv2 # Opencv can not read GIF image directly!
from PIL import Image

if __name__ == '__main__':
    # change data_root for different datasets.
    # First, we can use os.listdir() to get every image name.
    data_root = "../datasets"
    image_names = os.listdir(os.path.join(data_root, "train")) # return JPEG image names.
    # os.listdir() return a list, which includes the name of folder or file in a appointed foldr.
    # Do not include "." and "..".

    for filename in ["train", "train_masks"]:
        for image_name in image_names:
            # image_name
            
            if filename is "train":
                image_file = os.path.join(data_root, filename, image_name)
                image = Image.open(image_file).convert("L")
                image.save(os.path.join("../datasets/CarvanaImages", filename, image_name))

            if filename is "train_masks":
                image_file = os.path.join(data_root, filename, image_name[:-4] + "_mask.gif")
                image = Image.open(image_file).convert("L")
                image.save(os.path.join("../datasets/CarvanaImages", filename, 
                    image_name[:-4] + "_mask.jpg"))
