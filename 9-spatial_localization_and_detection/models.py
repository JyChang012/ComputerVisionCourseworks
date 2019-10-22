import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Softmax
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:2'

# Set device and memory use
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[-1], True)
# gpu = tf.config.experimental.list_logical_devices('GPU')

keras = tf.keras
K = keras.backend


def bbox_regressor(input_shape=(None, None, 3), output_class=5, logits_output=False, transfer=True, freeze_conv=False):
    # input shape is (128, 128, 3)
    """A bounding box regressor based on vgg16."""
    inputs = keras.Input(shape=input_shape)

    block1_conv1 = Conv2D(64, [3, 3], padding='same', activation='relu', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, [3, 3], padding='same', activation='relu', name='block1_conv2')(block1_conv1)
    block1_pool = MaxPool2D((2, 2), padding='same', name='block1_pool')(block1_conv2)

    block2_conv1 = Conv2D(128, [3, 3], padding='same', activation='relu', name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, [3, 3], padding='same', activation='relu', name='block2_conv2')(block2_conv1)
    block2_pool = MaxPool2D((2, 2), padding='same', name='block2_pool')(block2_conv2)

    block3_conv1 = Conv2D(256, [3, 3], padding='same', activation='relu', name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, [3, 3], padding='same', activation='relu', name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, [3, 3], padding='same', activation='relu', name='block3_conv3')(block3_conv2)
    block3_pool = MaxPool2D((2, 2), padding='same', name='block3_pool')(block3_conv3)

    block4_conv1 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block4_conv3')(block4_conv2)
    block4_pool = MaxPool2D((2, 2), padding='same', name='block4_pool')(block4_conv3)

    block5_conv1 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block5_conv3')(block5_conv2)
    block5_pool = MaxPool2D((2, 2), padding='same', name='block5_pool')(block5_conv3)
    flatten_feature = Flatten()(block5_pool)

    # Classification head
    gap = keras.layers.GlobalAveragePooling2D()(block5_pool)
    clf_fc1 = Dense(512, activation='relu', name='clf_fc1')(gap)
    clf_fc2 = Dense(512, activation='relu', name='clf_fc2')(clf_fc1)
    clf_fc3 = Dense(256, activation='relu', name='clf_fc3')(clf_fc2)
    clf_fc4 = Dense(256, activation='relu', name='clf_fc4')(clf_fc3)
    # clf_fc5 = Dense(256, activation='relu', name='clf_fc5')(clf_fc4)
    # clf_fc6 = Dense(128, activation='relu', name='clf_fc6')(clf_fc5)
    # clf_fc7 = Dense(128, activation='relu', name='clf_fc7')(clf_fc6)
    # clf_fc8 = Dense(128, activation='relu', name='clf_fc8')(clf_fc7)

    if logits_output:
        clf_output = Dense(output_class, name='clf_output')(clf_fc4)
    else:
        clf_output = Dense(output_class, activation='softmax', name='clf_output')(clf_fc4)

    # Box coordinates head
    box_fc1 = Dense(512, activation='relu', name='box_fc1')(flatten_feature)
    box_fc2 = Dense(512, activation='relu', name='box_fc2')(box_fc1)
    box_fc3 = Dense(256, activation='relu', name='box_fc3')(box_fc2)
    box_fc4 = Dense(128, activation='relu', name='box_fc4')(box_fc3)
    # box_fc5 = Dense(128, activation='relu', name='box_fc5')(box_fc4)
    # box_fc6 = Dense(256, activation='relu', name='box_fc6')(box_fc5)
    # box_fc7 = Dense(128, activation='relu', name='box_fc7')(box_fc6)
    # box_fc8 = Dense(128, activation='relu', name='box_fc8')(box_fc7)

    box_output = Dense(4,
                       # activation='relu',
                       name='box_output')(box_fc4)  # output (x, y, w, h)
    # TODO: other activations to restrict output?

    model = keras.Model(inputs=inputs, outputs=[clf_output, box_output])

    # transfer weights (for only conv kernel)
    if transfer:
        for layer in keras.applications.vgg16.VGG16().layers:
            name = layer.name
            if 'block' in name:
                model.get_layer(name).set_weights(layer.get_weights())

    if freeze_conv:
        for layer in model.layers:
            name = layer.name
            if 'block' in name:
                layer.trainable = False

    return model


if __name__ == '__main__':
    model = bbox_regressor((128, 128, 3))
    pass



