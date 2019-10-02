import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from BilinearUpSampling import BilinearUpSampling2D
import matplotlib.pyplot as plt
import os.path
from vis import voc_label_indices, build_colormap2label, batch_label_indices
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D, Add, Activation
import sys
import glob
from PIL import Image
# from predict import get_x, get_y

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

keras = tf.keras
VGG16 = keras.applications.vgg16.VGG16
vgg16 = keras.applications.vgg16
K = keras.backend
AUTOTUNE = tf.data.experimental.AUTOTUNE

# tf.enable_eager_execution()

# class fcn_8s_v2(keras.Model):
#
#     def __init__(self):
#         super().
# palette, color2class_map = make_palette(22)


# def voc_seg_loss(y_true, y_pred):

x_dir = './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/img/'
y_dir = './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/SegmentationClass/'


def voc_seg_acc(y_true, y_pred):  # Note y_true and y_pred always have the same shape, and x and y are automatically converted to the same ndim
    y_pred = K.argmax(y_pred, axis=-1)[..., tf.newaxis]
    y_true = K.cast(y_true, tf.int64)
    rst = K.equal(y_pred, y_true)
    return K.mean(rst)


def voc_seg_acc_v2(y_true, y_pred):  # Note y_true and y_pred always have the same shape, and x and y are automatically converted to the same ndim
    y_pred = K.reshape(y_pred, (-1, 21))
    y_true = K.reshape(y_true, (-1, 1))

    return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def voc_seg_loss(y_true, y_pred):
    # shape = K.shape(y_true)

    y_true = K.reshape(y_true, [-1, 1])
    y_pred = K.reshape(y_pred, [-1, 21])

    return K.sparse_categorical_crossentropy(y_true, y_pred)


def voc_seg_loss_v2(y_true, y_pred):
    y_true = K.one_hot(K.cast(K.reshape(y_true, (-1,)), tf.int32), 21)
    y_pred = K.reshape(y_pred, [-1, 21])

    rst = y_true * y_pred
    rst = K.sum(rst, axis=-1)
    return K.sum(K.log(rst))



def voc_label_indices_for_tf(colormap, colormap2label):
    """Map a RGB color to a label."""
    def fn(val):
        return tf.map_fn(lambda x: colormap2label[x], val, back_prop=False)

    colormap = tf.cast(colormap, tf.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return tf.map_fn(fn, idx, back_prop=False)


def preprocess_v4(name):
    x = tf.strings.join([x_dir, name, '.jpg'])
    y = tf.strings.join([y_dir, name, '.png'])

    x = process_path(x)
    y = process_path(y)

    x = preprocess_img(x)
    y = voc_label_indices_for_tf(y, tf.convert_to_tensor(build_colormap2label()))
    y = tf.image.resize_with_crop_or_pad(y[..., np.newaxis], 512, 512)

    return x, y


def preprocess_v3(x, y):
    # x = example[0]
    # y = example[1]

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


def batch_pad(imgs):
    rst = []
    if imgs[0].ndim == 2:
        for img in imgs:
            rst.append(tf.reshape(tf.image.resize_with_crop_or_pad(img[..., np.newaxis], 512, 512), [512, 512]))
    else:
        for img in imgs:
            rst.append(tf.image.resize_with_crop_or_pad(img, 512, 512))

    return rst


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


def batch_preprocess_img(imgs):
    rst = []
    for img in imgs:
        rst.append(preprocess_img(img))
    return rst


def preprocess_img(img):
    img = tf.to_float(img)
    img = keras.applications.vgg16.preprocess_input(img)
    return tf.image.resize_with_crop_or_pad(img, 512, 512)


def fcn_8s_exp(input_shape=(None, None, 3), output_classes=21):
    inputs = keras.Input(shape=input_shape)
    vgg16 = VGG16()
    block5_pool_output = vgg16.get_layer('block5_pool').output
    block4_pool_output = vgg16.get_layer('block4_pool').output
    block3_pool_output = vgg16.get_layer('block3_pool').output

    fc1_output = keras.layers.Conv2D(4096, (7, 7), activation='relu', padding='same',
                                     name='fc1')(block5_pool_output)
    fc2_output = keras.layers.Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(fc1_output)

    fc2_score = keras.layers.Conv2D(output_classes, (1, 1), padding='same', name='fc2_score')(fc2_output)
    fc2_4x = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear', trainable=False, name='fc2_4x')(fc2_score)

    # fc2_2x = keras.layers.Conv2DTranspose(output_classes, (2, 2), strides=(2, 2), padding='same', use_bias=False, name='fc2_2x')(fc2_score)
    # fc2_4x = keras.layers.Conv2DTranspose(output_classes, (2, 2), strides=(2, 2), padding='same', use_bias=False, name='fc2_4x')(fc2_2x)

    # fc2_4x = BilinearUpSampling2D(size=(4, 4), trainable=False, name='fc2_4x')(fc2_score)

    pool4_score = keras.layers.Conv2D(output_classes, (1, 1), padding='same', name='pool4_score')(block4_pool_output)
    pool4_2x = keras.layers.UpSampling2D((2, 2), interpolation='bilinear', trainable=False,
                                         name='pool4_2x')(pool4_score)
    # pool4_2x = keras.layers.Conv2DTranspose(output_classes, (2, 2), strides=(2, 2), padding='same', use_bias=False, name='pool4_2x')(pool4_score)
    # pool4_2x = BilinearUpSampling2D((2, 2), trainable=False, name='pool4_2x')(pool4_score)

    pool3_score = keras.layers.Conv2D(output_classes, (1, 1), padding='same', name='poo3_score')(block3_pool_output)

    _sum = keras.layers.Add()([fc2_4x, pool4_2x, pool3_score])
    sum_8x = keras.layers.UpSampling2D((8, 8), interpolation='bilinear', trainable=False, name='sum_8x')(_sum)

    # sum_2x = keras.layers.Conv2DTranspose(output_classes, (2, 2), strides=(2, 2), padding='same', use_bias=False,
    #                                       name='sum_2x')(_sum)
    # sum_4x = keras.layers.Conv2DTranspose(output_classes, (2, 2), strides=(2, 2), padding='same', use_bias=False,
    #                                       name='sum_4x')(sum_2x)
    # sum_8x = keras.layers.Conv2DTranspose(output_classes, (2, 2), strides=(2, 2), padding='same', use_bias=False,
    #                                       name='sum_8x')(sum_4x)

    # sum_8x = BilinearUpSampling2D((8, 8), trainable=False, name='sum_8x')(_sum)
    outputs = keras.layers.Activation('softmax')(sum_8x)

    model = keras.Model(inputs=vgg16.layers[1].input, outputs=outputs)

    # transfer weights
    fc1_weights = vgg16.get_layer('fc1').get_weights()
    fc1_weights[0] = fc1_weights[0].reshape((7, 7, 512, 4096))
    model.get_layer('fc1').set_weights(fc1_weights)

    fc2_weights = vgg16.get_layer('fc2').get_weights()
    fc2_weights[0] = fc2_weights[0].reshape((1, 1, 4096, 4096))
    model.get_layer('fc2').set_weights(fc2_weights)

    # change padding setting for pooling layer
    # for layer in model.layers:
    #     # if 'pool' in layer.name:
    #     if type(layer) is keras.layers.Conv2D:
    #         layer.padding = 'valid'

    for layer in model.layers:
        if 'pool' in layer.name:
            layer.padding = 'same'
    #
    # x = keras.layers.ZeroPadding2D(100)(inputs)
    # x = model(x)
    # outputs = keras.layers.Cropping2D(100)(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def fcn_8s_v3(input_shape=(None, None, 3), output_classes=21):
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
    outputs = Activation('softmax', name='softmax')(add_8x)

    model = keras.Model(inputs=inputs, outputs=outputs)

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

    for layer in model.layers:
        if 'conv' in layer.name and int(layer.name[5]) <= 3:
            layer.trainable = False

    return model


def preprocess_v2():
    # train_file_names = np.loadtxt(
    #     './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', dtype=np.str)
    trainval_file_names = np.loadtxt(
        './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)
    # val_file_names = np.loadtxt(
    #     './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', dtype=np.str)
    # segvall11 = np.loadtxt('./fcn.berkeleyvision.org/data/pascal/seg11valid.txt', dtype=np.str)

    x_trainval_dir = []
    y_trainval_dir = []
#
    for name in trainval_file_names:
        x_trainval_dir.append(os.path.join('./fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/img', name + '.jpg'))
        y_trainval_dir.append(os.path.join('./fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/SegmentationClass',
                                           name + '.png'))

    # name_list = tf.data.Dataset.list_files(trainval_file_names, seed=12)
    # x_img_ori = x_list.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # x_img = x_img_ori.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    x_img_ori = [np.array(keras.preprocessing.image.load_img(path)) for path in x_trainval_dir]
    x_img = batch_preprocess_img(x_img_ori)
    y_img_ori = [np.array(keras.preprocessing.image.load_img(path)) for path in y_trainval_dir]
    y_img_label = batch_label_indices(y_img_ori)

    # data = tf.data.Dataset.from_tensor_slices((x_img_ori, y_img_label))

    y_img_label = batch_pad(y_img_label)

    # x_img = tf.convert_to_tensor(x_img)
    # y_img_label = tf.convert_to_tensor(y_img_label)

    # data = tf.data.Dataset.from_tensor_slices((x_img, y_img_label))

    return x_img, y_img_label


    # data = name_list.map(process_name, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # y_img = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_img_list))

    # y_img_ori = y_list.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # y_img = y_img_ori.map(color2class, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # i = 0
    # for img1, img2 in zip(x_img_ori, y_img_ori):
    #     plt.imshow(img1)
    #     plt.show()
    #     plt.imshow(img2)
    #     plt.show()
    #     i += 1
    #     if i > 8:
    #         break

    # data = tf.data.Dataset.from_tensor_slices([x_data, y_data])

    # return data


def get_data_v3():
    trainval_file_names = np.loadtxt(
        './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)

    x_trainval_dir = []
    y_trainval_dir = []
    #
    for name in trainval_file_names:
        x_trainval_dir.append(os.path.join('./fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/img', name + '.jpg'))
        y_trainval_dir.append(os.path.join('./fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/SegmentationClass',
                                           name + '.png'))

    data = tf.data.Dataset.from_tensor_slices((x_trainval_dir, y_trainval_dir))
    data = data.shuffle(1000)
    data = data.map(preprocess_v3, num_parallel_calls=AUTOTUNE)
    # data = data.cache('/home/changjy/tf_cache')
    return data, len(x_trainval_dir)


def get_data_v4():
    trainval_file_names = np.loadtxt(
        './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)

    data = tf.data.Dataset.from_tensor_slices(trainval_file_names).shuffle(1000)
    data = data.map(preprocess_v4, AUTOTUNE)
    return data, len(trainval_file_names)


def get_data_v5(shuffle=True, split=0.2):
    trainval_file_names = np.loadtxt(
        './fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)
    if split is False:
        data = tf.data.Dataset.from_tensor_slices(trainval_file_names)
        if shuffle is True:
            data = data.shuffle(1000)
        data = data.map(preprocess_v4, num_parallel_calls=AUTOTUNE)
        return data, len(trainval_file_names)
    else:
        n = int(split * len(trainval_file_names))
        if shuffle is False:
            data = tf.data.Dataset.from_tensor_slices(trainval_file_names).map(preprocess_v4,
                                                                               num_parallel_calls=AUTOTUNE)
            train = data.take(n)
            val = data.skip(n)

            return train, val, len(trainval_file_names)
        else:
            data = tf.data.Dataset.from_tensor_slices(trainval_file_names).shuffle(1000).map(preprocess_v4, num_parallel_calls=AUTOTUNE)

            val = data.take(n)
            train = data.skip(n)

            return train, val, len(trainval_file_names)


def test_model_without_val():

    data, n = get_data_v4()
    data = data.batch(1).repeat(10).prefetch(AUTOTUNE)
    callbacks = [keras.callbacks.TensorBoard('./logs/not_val')]
    model = fcn_8s_v3()
    # print(model.layers[2].get_config())
    model.compile(optimizer=keras.optimizers.Adam(5e-7), loss='sparse_categorical_crossentropy',
                  metrics=[voc_seg_acc])

    history = model.fit(data, epochs=1000, steps_per_epoch=20, callbacks=callbacks)
    model.save('./models/no_val/first_model_train2.h5')
    # pred = model.predict(data.map(get_x, AUTOTUNE))
    # pred = np.argmax(pred, axis=-1)

    pass

    # x = np.random.random((7, 512, 256, 3)).astype(np.float32)


def test_model():
    callbacks = [keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                 keras.callbacks.TensorBoard('./logs/val')]
    model = fcn_8s_v3()
    model.compile(optimizer=keras.optimizers.Adam(5e-7), loss='sparse_categorical_crossentropy',
                  metrics=[voc_seg_acc_v2])

    data, n = get_data_v4()
    train_data = data.take(n - 24).batch(1).repeat(100)
    val_data = data.skip(n - 24).batch(1)

    # train_data, val_data, n = get_data_v5()
    # train_data = train_data.batch(1).repeat(100)
    # val_data = val_data.batch(1)

    history = model.fit(train_data, epochs=10000, steps_per_epoch=20, validation_data=val_data, callbacks=callbacks,
                        validation_steps=10)
    model.save('./models/val/first_model.h5')
    pass


def choose():
    if sys.argv[0] == '--with_val':
        test_model()
    elif sys.argv[0] == '--no_val':
        test_model()
    else:
        print('no valid param!')


if __name__ == '__main__':
    test_model_without_val()
