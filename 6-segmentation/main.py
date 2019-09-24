import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

keras = tf.keras


class FCN_8s(keras.layers.Layer):
    def __init__(self, num_classes=21):
        super().__init__(name='FCN_8s')
        self.num_classes = num_classes
        self.conv1 = keras.layers.Conv2D([3, 3], 2, padding='same', activation='relu')
        self.pool1 = keras.layers.MaxPool2D([2, 2])
        self.conv2 = keras.layers.Conv2D([3, 3], 2, padding='same', activation='relu')
        self.pool2 = keras.layers.MaxPool2D([2, 2])
        self.conv3 = keras.layers.Conv2D([3, 3], 3, padding='same', activation='relu')
        self.pool3 = keras.layers.MaxPool2D([2, 2])
        self.conv4 = keras.layers.Conv2D([3, 3], 3, padding='same', activation='relu')
        self.pool4 = keras.layers.MaxPool2D([2, 2])
        self.conv5 = keras.layers.Conv2D([3, 3], 3, padding='same', activation='relu')
        self.pool5 = keras.layers.MaxPool2D([2, 2])
        self.conv6 = keras.layers.Conv2D([3, 3], 2, padding='same', activation='relu')
        self.conv7 = keras.layers.Conv2D([3, 3], self.num_classes, padding='same', activation='relu')

        self.upsampling_conv7 = keras.layers.UpSampling2D([4, 4])  # TODO: look up the api of up sampling layers
        self.output_conv7 = keras.layers.Conv2D([1, 1], self.num_classes, padding='same', activation='relu')
        self.upsampling_pool4 = keras.layers.UpSampling2D([2, 2])
        self.output_pool4 = keras.layers.Conv2D([1, 1], self.num_classes, padding='same', activation='relu')
        self.output_pool3 = keras.layers.Conv2D([1, 1], self.num_classes, padding='same', activation='relu')

        self.output_x8 = keras.layers.UpSampling2D([8, 8])
        self.softmax = keras.layers.Softmax()  # TODO: ??

    def call(self, inputs, **kwargs):
        c1 = self.conv1(inputs)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        p5 = self.pool5(c5)
        c6 = self.conv6(p5)
        c7 = self.conv7(c6)

        c7_x4 = self.upsampling_conv7(c7)
        c7_o = self.output_conv7(c7_x4)

        p4_x2 = self.upsampling_pool4(p4)
        p4_o = self.output_pool4(p4_x2)

        p3_o = self.output_pool3(p3)

        _sum = c7_o + p4_o + p3_o
        x8 = self.output_x8(_sum)
        o = self.softmax(x8)

        return o

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)





