# -*- coding:utf-8 -*-
"""
Read TFRecords file.
"""
import os
import tensorflow as tf
import numpy as np
import scipy.misc

class Read_TFRecords(object):
    def __init__(self, filename, batch_size=64,
        image_h=256, image_w=256, image_c=1, num_threads=8, capacity_factor=3, min_after_dequeue=1000):
        '''
        filename: TFRecords file path.
        num_threads: TFRecords file load thread.
        capacity_factor: capacity.
        '''
        self.filename = filename
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.num_threads = num_threads
        self.capacity_factor = capacity_factor
        self.min_after_dequeue = min_after_dequeue

    def read(self):
        # read a TFRecords file, return tf.train.batch/tf.train.shuffle_batch object.
        reader = tf.TFRecordReader()
        
        filename_queue = tf.train.string_input_producer([self.filename])
        key, serialized_example = reader.read(filename_queue)
        
        # Now, will have two Features.
        features = tf.parse_single_example(serialized_example,
            features={
                "image_raw": tf.FixedLenFeature([], tf.string),
                "image_label": tf.FixedLenFeature([], tf.string),
            })
       
        image_raw = tf.image.decode_jpeg(features["image_raw"], channels=self.image_c, 
            name="decode_image")
        image_label = tf.image.decode_jpeg(features["image_label"], channels=self.image_c, 
            name="decode_image")
        # not need Crop and other random augmentations.
        # image resize and transform type.
        # Utilize tf.gfile.FastGFile() to generate TFRecords file, in this way, it could use resize_images directly.
        if self.image_h is not None and self.image_w is not None:
            image_raw = tf.image.resize_images(image_raw, [self.image_h, self.image_w], 
                method=tf.image.ResizeMethod.BICUBIC)
            image_label = tf.image.resize_images(image_label, [324, 324], 
                method=tf.image.ResizeMethod.BICUBIC)
            # TODO: The shape of image masks. Refer to the Unet in model.py, the output image is
            # 324 * 324 * 1. But is not good.

        image_raw = tf.cast(image_raw, tf.float32) / 255.0 # convert to float32
        image_label = tf.cast(image_label, tf.float32) / 255.0 # convert to float32

        # tf.train.batch/tf.train.shuffle_batch object.
        # Using asynchronous queues
        input_data, input_masks = tf.train.shuffle_batch([image_raw, image_label],
            batch_size=self.batch_size,
            capacity=self.min_after_dequeue + self.capacity_factor * self.batch_size,
            min_after_dequeue=self.min_after_dequeue,
            num_threads=self.num_threads,
            name='images')
        
        return input_data, input_masks # return list or dictionary of tensors.
