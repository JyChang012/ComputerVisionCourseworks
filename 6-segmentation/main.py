import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from transform import voc_label_indices, build_colormap2label, batch_label_indices
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D, Add, Activation
import sys
import glob
from PIL import Image


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2"

keras = tf.keras
VGG16 = keras.applications.vgg16.VGG16
vgg16 = keras.applications.vgg16
K = keras.backend
AUTOTUNE = tf.data.experimental.AUTOTUNE

# tf.enable_eager_execution()


x_dir = './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/img/'
y_dir = './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/SegmentationClass/'


def voc_label_indices_for_tf(y, colormap2label):
    """Map a RGB color image to a label."""
    def fn(val):
        return tf.map_fn(lambda x: colormap2label[x], val, back_prop=False)

    y = tf.cast(y, tf.int32)
    idx = ((y[:, :, 0] * 256 + y[:, :, 1]) * 256
           + y[:, :, 2])
    return tf.map_fn(fn, idx, back_prop=False)


def preprocess_with_random_crop(aug_n=10):
    def preprocess(name):
        x = tf.strings.join([x_dir, name, '.jpg'])
        y = tf.strings.join([y_dir, name, '.png'])

        x = process_path(x)
        y = process_path(y)

        x = keras.applications.vgg16.preprocess_input(tf.cast(x, tf.float32))
        y = voc_label_indices_for_tf(y, tf.convert_to_tensor(build_colormap2label()))[..., tf.newaxis]
        y = tf.cast(y, tf.float32)

        img = tf.concat([x, y], axis=2)
        rst = tf.image.random_crop(img, (224, 224, 4))
        aug_x = [rst[:, :, :-1]]
        aug_y = [rst[:, :, -1][..., tf.newaxis]]

        for _ in range(aug_n-1):
            rst = tf.image.random_crop(img, (224, 224, 4))
            aug_x.append(rst[:, :, :-1])
            aug_y.append(rst[:, :, -1][..., tf.newaxis])

        return tf.data.Dataset.from_tensor_slices((aug_x, aug_y))

    return preprocess


def preprocess(name):
    x = tf.strings.join([x_dir, name, '.jpg'])
    y = tf.strings.join([y_dir, name, '.png'])

    x = process_path(x)
    y = process_path(y)

    x = preprocess_img(x)
    y = voc_label_indices_for_tf(y, tf.convert_to_tensor(build_colormap2label()))
    y = tf.image.resize_with_crop_or_pad(y[..., np.newaxis], 512, 512)

    return x, y


def process_name(name=''):
    x = process_path(tf.strings.join([x_dir, name+'.jpg']))
    x = preprocess_img(x)
    y = np.array(keras.preprocessing.image.load_img(tf.strings.join([y_dir, name+'.png'])))
    y = voc_label_indices(y, build_colormap2label()).astype(np.int32)
    y = pad_and_crop(y)
    return x, y


def pad_and_crop(img):
    if img.ndim == 3:
        return tf.image.resize_with_crop_or_pad(img, 512, 512)
    elif img.ndim == 2:
        return tf.reshape(tf.image.resize_with_crop_or_pad(img[..., np.newaxis], 512, 512), [512, 512])


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def preprocess_img(img):
    img = tf.to_float(img)
    img = keras.applications.vgg16.preprocess_input(img)
    return tf.image.resize_with_crop_or_pad(img, 512, 512)


def fcn_8s(input_shape=(None, None, 3), output_classes=21):
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

    fc1 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(block5_pool)
    fc2 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(fc1)
    fc2_score = Conv2D(output_classes, (1, 1), padding='same', name='fc2_score')(fc2)
    fc2_4x = UpSampling2D(size=(4, 4), interpolation='bilinear', trainable=False, name='fc2_4x')(fc2_score)

    pool4_score = Conv2D(output_classes, (1, 1), padding='same', name='pool4_score')(block4_pool)
    pool4_2x = UpSampling2D((2, 2), interpolation='bilinear', trainable=False, name='pool4_2x')(pool4_score)

    pool3_score = Conv2D(output_classes, (1, 1), padding='same', name='poo3_score')(block3_pool)

    add = Add()([fc2_4x, pool4_2x, pool3_score])
    add_8x = UpSampling2D((8, 8), interpolation='bilinear', trainable=False, name='add_8x')(add)
    # outputs = Activation('softmax', name='softmax')(add_8x)

    # model = keras.Model(inputs=inputs, outputs=outputs)
    model = keras.Model(inputs=inputs, outputs=add_8x)

    # transfer weights
    for layer in VGG16().layers:
        name = layer.name
        if 'block' in name:
            model.get_layer(name).set_weights(layer.get_weights())
        elif name == 'fc1':
            fc1_weights = layer.get_weights()
            fc1_weights[0] = fc1_weights[0].reshape((7, 7, 512, 4096))
            model.get_layer('fc1').set_weights(fc1_weights)
        elif name == 'fc2':
            fc2_weights = layer.get_weights()
            fc2_weights[0] = fc2_weights[0].reshape((1, 1, 4096, 4096))
            model.get_layer('fc2').set_weights(fc2_weights)

    # for layer in model.layers:
    #     if 'conv' in layer.name and int(layer.name[5]) <= 3:
    #         layer.trainable = False

    return model


def get_data():
    trainval_file_names = np.loadtxt(
        './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)

    data = tf.data.Dataset.from_tensor_slices(trainval_file_names).shuffle(1000)
    data = data.map(preprocess, AUTOTUNE)
    return data, len(trainval_file_names)


def get_data_random_crop(aug_n=10):
    trainval_file_names = np.loadtxt(
        './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)
    data = tf.data.Dataset.from_tensor_slices(trainval_file_names).shuffle(1000)

    data = data.interleave(preprocess_with_random_crop(aug_n), cycle_length=4, num_parallel_calls=AUTOTUNE)

    return data, len(trainval_file_names) * aug_n


def test_model_without_val():

    data, n = get_data()
    # data = data.repeat(1000).batch(1).prefetch(AUTOTUNE)
    data = data.batch(2).prefetch(AUTOTUNE)
    callbacks = [keras.callbacks.TensorBoard('./logs/not_val/new')]
    model = fcn_8s()
    # print(model.layers[2].get_config())
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # 1e-5?
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    history = model.fit(data, epochs=100, callbacks=callbacks)
    # history = model.fit(data, epochs=250)
    model.save('./models/no_val/first_model_train3.h5')
    # pred = model.predict(data.map(get_x, AUTOTUNE))
    # pred = np.argmax(pred, axis=-1)

    pass

    # x = np.random.random((7, 512, 256, 3)).astype(np.float32)


def test_model_random_crop():
    data, n = get_data_random_crop(10)
    data = data.batch(2).prefetch(AUTOTUNE)
    # callbacks =
    model = fcn_8s()

    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # 1e-5?
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    history = model.fit(data, epochs=100, steps_per_epoch=50, shuffle=True)


def test_model():
    callbacks = [keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                 keras.callbacks.TensorBoard('./logs/val')]
    model = fcn_8s()
    model.compile(optimizer=keras.optimizers.Adam(1e-6), loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    data, n = get_data()
    # repeat before batch for better performance
    train_data = data.take(int(0.8 * n)).repeat(100).batch(1).prefetch(AUTOTUNE)
    val_data = data.skip(int(0.8 * n)).batch(1).prefetch(AUTOTUNE)

    # train_data, val_data, n = get_data_v5()
    # train_data = train_data.batch(1).repeat(100)
    # val_data = val_data.batch(1)

    history = model.fit(train_data, epochs=10000, steps_per_epoch=20, validation_data=val_data, callbacks=callbacks)
    model.save('./models/val/first_model_t2.h5')
    pass


def choose():
    if sys.argv[1] == '--with_val':
        test_model()
    elif sys.argv[1] == '--no_val':
        test_model_without_val()
    else:
        print('no valid param!')


if __name__ == '__main__':
    test_model_random_crop()
