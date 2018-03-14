# -*- coding:utf-8 -*-
"""
Generate TFRecords file, for training.
"""
import os
# import glob # Can use os.listdir(data_dir) replace glob.glob(os.path.join(data_dir, "*.jpg"))
# to get every image name, do not include path.
import tensorflow as tf

if __name__ == '__main__':
    # change data_root for different datasets.
    # First, we can use os.listdir() to get every image name.
    data_root = "../datasets/CarvanaImages"
    image_names = os.listdir(os.path.join(data_root, "train")) # return JPEG image names.
    # os.listdir() return a list, which includes the name of folder or file in a appointed foldr.
    # Do not include "." and "..".

    # TFRecordWriter, dump to tfrecords file
    # TFRecord file name, change save_name for different datasets.
    # Create one proto buffer, then add two Features.
    if not os.path.exists(os.path.join("../datasets", "tfrecords")):
        os.makedirs(os.path.join("../datasets", "tfrecords"))
    writer = tf.python_io.TFRecordWriter(os.path.join("../datasets", "tfrecords", 
        "Carvana.tfrecords"))

    for image_name in image_names:
        # image_name
        image_raw_file = os.path.join(data_root, "train", image_name)
        image_label_file = os.path.join(data_root, "train_masks", 
            image_name[:-4] + "_mask.jpg")

        # The first method to load image.
        '''
        image_raw = Image.open(image_file) # It image is RGB, then mode=RGB; otherwise, mode=L.
        # reszie image. In reading the TFRecords file, if you want resize the image, you could put image 
        # height and width into the Feature.
        # In this way, when reading the TFRecords file, it can use width and height.
        width = image_raw.size[0]
        height = image_raw.size[1]
        # put image height and width into the Feature.
        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))

        image_raw = image_raw.tobytes()
        # Transform image to byte.
        '''

        # Second method to load image.
        image_raw = tf.gfile.FastGFile(image_raw_file, 'rb').read() # image data type is string. 
        # read and binary.
        image_label = tf.gfile.FastGFile(image_label_file, 'rb').read()

        # write bytes to Example proto buffer.
        example = tf.train.Example(features=tf.train.Features(feature={
            "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            "image_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label]))
            }))
        
        writer.write(example.SerializeToString()) # Serialize To String
    
    writer.close()
