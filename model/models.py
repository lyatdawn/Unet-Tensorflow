# -*- coding:utf-8 -*-
"""
Generator and Discriminator network.
"""
import tensorflow as tf
import utils

# Define Unet, you can refer to http://blog.csdn.net/u014722627/article/details/60883185 or 
# https://github.com/zhixuhao/unet
def Unet(name, in_data, reuse=False):
    # Not use BatchNorm or InstanceNorm.
    assert in_data is not None
    with tf.variable_scope(name, reuse=reuse):
        # Conv1 + Crop1
        conv1_1 = tf.layers.conv2d(in_data, 64, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0)) # Use Xavier init.
        # Arguments: inputs, filters, kernel_size, strides((1, 1)), padding(VALID). 
        # Appoint activation, use_bias, kernel_initializer, bias_initializer=tf.zeros_initializer().
        # In Keras's implement, kernel_initializer is he_normal, i.e. 
        # mean = 0.0, stddev = sqrt(2 / fan_in).
        '''
        In Tensorflow, you can use tf.keras.initializers.he_normal to implement he_normal, 
        but tf.keras.initializers.he_normal is a function, not a class. So we can not use it
        directly. But, if you look at the implementation of he_normal, it is actually use
        tf.variance_scaling_initializer.

        tf.variance_scaling_initializer class. Initializer capable of adapting its scale to the shape of weights tensors.
        Args:
        scale: Scaling factor (positive float). default 1.
        mode: One of "fan_in", "fan_out", "fan_avg". default 'fan_in'.
        distribution: Random distribution to use. One of "normal", "uniform". default "normal".
        seed: A Python integer. Used to create random seeds.

        With distribution="normal", from a normal distribution mean zero, with stddev = sqrt(scale / n), where n is
        the number of input units in the weight tensor, if mode = "fan_in".
        the number of output units, if mode = "fan_out".
        average of the numbers of input and output units, if mode = "fan_avg".

        With distribution="uniform", from a uniform distribution within [-limit, limit], 
        with limit = sqrt(3 * scale / n).

        So, we can set kernel_initializer = tf.variance_scaling_initializer(scale=2.0)
        '''

        conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        crop1 = tf.keras.layers.Cropping2D(cropping=((90, 90), (90, 90)))(conv1_2)
        '''
        Use Tensorflow and Keras, in Tensorflow there are some Keras's APIS, you can use 
        Tensorflow and Keras at the same time. For example:
        x = tf.keras.layers.Dense(128, activation='relu')(input)

        tf.keras.layers.Cropping2D, class Cropping layer for 2D input. Arguments:
        cropping: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            int: cropping is applied to width and height.
            (symmetric_height_crop, symmetric_width_crop): cropping values for height and width
            ((top_crop, bottom_crop), (left_crop, right_crop))
        First, define a object of class, then use object(data) to forward.
        '''

        # MaxPooling1 + Conv2 + Crop2
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple), 
        # padding='valid'.
        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        crop2 = tf.keras.layers.Cropping2D(cropping=((41, 41), (41, 41)))(conv2_2)

        # MaxPooling2 + Conv3 + Crop3
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple), 
        # padding='valid'.
        conv3_1 = tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        crop3 = tf.keras.layers.Cropping2D(cropping=((16, 17), (16, 17)))(conv3_2)

        # MaxPooling3 + Conv4 + Drop4 + Crop4
        pool3 = tf.layers.max_pooling2d(conv3_2, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple), 
        # padding='valid'.
        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        drop4 = tf.layers.dropout(conv4_2)
        # Arguments: inputs, rate=0.5.
        crop4 = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(drop4)

        # MaxPooling4 + Conv5 + Crop5
        pool4 = tf.layers.max_pooling2d(drop4, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple), 
        # padding='valid'.
        conv5_1 = tf.layers.conv2d(pool4, 1024, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv5_2 = tf.layers.conv2d(conv5_1, 1024, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        drop5 = tf.layers.dropout(conv5_2)

        # Upsampling6 + Conv + Merge6
        up6_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop5)
        '''
        Class UpSampling2D, Upsampling layer for 2D inputs. Arguments:
        size: int, or tuple of 2 integers. The upsampling factors for rows and columns.
        '''
        up6 = tf.layers.conv2d(up6_1, 512, 2, padding="SAME", activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        merge6 = tf.concat([crop4, up6], axis=3) # concat channel
        # values: A list of Tensor objects or a single Tensor.

        # Conv6 + Upsampling7 + Conv + Merge7
        conv6_1 = tf.layers.conv2d(merge6, 512, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))

        up7_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2)
        up7 = tf.layers.conv2d(up7_1, 256, 2, padding="SAME", activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        merge7 = tf.concat([crop3, up7], axis=3) # concat channel

        # Conv7 + Upsampling8 + Conv + Merge8
        conv7_1 = tf.layers.conv2d(merge7, 256, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv7_2 = tf.layers.conv2d(conv7_1, 256, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))

        up8_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        up8 = tf.layers.conv2d(up8_1, 128, 2, padding="SAME", activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        merge8 = tf.concat([crop2, up8], axis=3) # concat channel

        # Conv8 + Upsampling9 + Conv + Merge9
        conv8_1 = tf.layers.conv2d(merge8, 128, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv8_2 = tf.layers.conv2d(conv8_1, 128, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))

        up9_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_2)
        up9 = tf.layers.conv2d(up9_1, 64, 2, padding="SAME", activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        merge9 = tf.concat([crop1, up9], axis=3) # concat channel

        # Conv9
        conv9_1 = tf.layers.conv2d(merge9, 64, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv9_2 = tf.layers.conv2d(conv9_1, 64, 3, activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        conv9_3 = tf.layers.conv2d(conv9_2, 2, 3, padding="SAME", activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))

        # Conv10
        conv10 = tf.layers.conv2d(conv9_3, 1, 1,
            kernel_initializer = tf.contrib.layers.xavier_initializer())
            # kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        # 1 channel.

    return conv10