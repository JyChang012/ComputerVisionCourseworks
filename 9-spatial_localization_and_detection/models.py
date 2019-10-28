import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:2'

# Set device and memory use
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[-1], True)
# gpu = tf.config.experimental.list_logical_devices('GPU')

keras = tf.keras
K = keras.backend


def bbox_regressor(input_shape=(None, None, 3), output_class=5, logits_output=False, freeze_conv=False):
    # input shape is (128, 128, 3)
    """A simpler implementation of bbox regressor."""
    vgg16 = keras.applications.vgg16.VGG16(include_top=False, input_shape=input_shape)
    flatten_feature = Flatten()(vgg16.outputs[0])

    # Classification head
    gap = GlobalAveragePooling2D()(vgg16.outputs[0])
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

    model = keras.Model(inputs=vgg16.inputs[0], outputs=[clf_output, box_output])

    # transfer weights (for only conv kernel)

    if freeze_conv:
        for layer in model.layers:
            name = layer.name
            if 'block' in name:
                layer.trainable = False

    return model


if __name__ == '__main__':
    model = bbox_regressor((128, 128, 3))
    pass



