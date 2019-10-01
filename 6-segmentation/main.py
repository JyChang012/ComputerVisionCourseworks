import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from BilinearUpSampling import BilinearUpSampling2D
import matplotlib.pyplot as plt
import os.path
from vis import voc_label_indices, build_colormap2label, batch_label_indices
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D, Add, Activation
import glob
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def voc_seg_metrics(y_true, y_predict):
    shape = tf.shape(y_true)
    num_of_pix = shape[0] * shape[1]
    y_predict = tf.argmax(y_predict, axis=-1, output_type=tf.int32)
    rst = (y_predict == y_true)
    return np.sum(rst) / num_of_pix

def voc_label_indices_for_tf(colormap, colormap2label):
    """Map a RGB color to a label."""
    def fn(val):
        return tf.map_fn(lambda x: colormap2label[x], val, back_prop=False)

    colormap = tf.cast(colormap, tf.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return tf.map_fn(fn, idx, back_prop=False)


def preprocess_v3(x, y):
    # x = example[0]
    # y = example[1]

    x = process_path(x)
    y = process_path(y)

    x = preprocess_img(x)
    y = voc_label_indices_for_tf(y, tf.convert_to_tensor(build_colormap2label()))
    y = tf.reshape(tf.image.resize_with_crop_or_pad(y[..., np.newaxis], 512, 512), [512, 512])

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


def fcn_8s(input_shape=None, output_classes=21):
    inputs = keras.Input(shape=input_shape)
    layers = dict()
    for layer in vgg16.layers:
        if 'conv' in layer.name:
            layers[layer.name] = keras.layers.Conv2D.from_config(layer)
        elif 'pool' in layer.name:
            layers[layer.name] = keras.layers.MaxPool2D.from_config(layer)

    x = layers['block1_conv1'](inputs)
    x = layers['block1_conv2'](x)
    x = layers['block1_pool'](x)
    x = layers['block2_conv1'](x)
    x = layers['block2_conv2'](x)
    x = layers['block2_pool'](x)
    x = layers['block3_conv1'](x)
    x = layers['block3_conv2'](x)
    x = layers['block3_conv3'](x)
    block_pool3_output = layers['block3_pool'](x)

    x = layers['block4_conv1'](block_pool3_output)
    x = layers['block4_conv2'](x)
    x = layers['block4_conv3'](x)
    block_pool4_output = layers['block4_pool'](x)

    x = layers['block5_conv1'](block_pool4_output)
    x = layers['block5_conv2'](x)
    x = layers['block5_conv3'](x)
    block_pool5_output = layers['block5_pool'](x)
    pass


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

    pool4_score = keras.layers.Conv2D(output_classes, (1, 1), padding='same', name='pool4_score')(block4_pool)
    pool4_2x = keras.layers.UpSampling2D((2, 2), interpolation='bilinear', trainable=False,
                                         name='pool4_2x')(pool4_score)

    pool3_score = keras.layers.Conv2D(output_classes, (1, 1), padding='same', name='poo3_score')(block3_pool)

    add = Add()([fc2_4x, pool4_2x, pool3_score])
    add_8x = keras.layers.UpSampling2D((8, 8), interpolation='bilinear', trainable=False, name='add_8x')(add)
    outputs = Activation('softmax')(add_8x)

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

    return model


def preprocess():
    img_generator = keras.preprocessing.image.ImageDataGenerator(fill_mode='constant', cval=0)
    x_iter = img_generator.flow_from_directory('./fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/',
                                               classes=['img'],
                                               target_size=(512, 512), class_mode=None, interpolation='bilinear')
    class_generator = keras.preprocessing.image.ImageDataGenerator(fill_mode='constant', cval=0)
    y_iter = img_generator.flow_from_directory('./fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/',
                                               classes=['SegmentationClass'], target_size=(512, 512), class_mode=None,
                                               interpolation='bilinear')

    x = next(iter(x_iter))
    pass


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
    data = data.map(preprocess_v3, num_parallel_calls=AUTOTUNE)
    # data = data.cache('/home/changjy/tf_cache')
    return data, len(x_trainval_dir)


def test_model():
    callbacks = [keras.callbacks.EarlyStopping(patience=30), keras.callbacks.TensorBoard()]
    model = fcn_8s_v3()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=[voc_seg_metrics])

    # x, y = preprocess_v2()
    # data = tf.data.Dataset.from_tensor_slices((x[:100], y[:100])).cache('/home/changjy/tf_cache')
    data, n = get_data_v3()
    train_data = data.take(int(0.7 * n))
    val_data = data.skip(int(0.7 * n))
    # i = 0
    # for _ in data:
    #     i += 1
    # print(i)
    # data = data.shuffle(100)
    train_data = train_data.repeat(100).batch(1)
    val_data = val_data.repeat(100).batch(1)
    # data.prefetch(3)
    # model.fit(tf.convert_to_tensor(data[0]), tf.convert_to_tensor(data[1]), batch_size=2, epochs=1000)
    history = model.fit(train_data, epochs=100, steps_per_epoch=20, validation_data=val_data, callbacks=callbacks)
    model.save('./models/first_model.h5')

    pass

    # x = np.random.random((7, 512, 256, 3)).astype(np.float32)


#
# y = model(x)
# print(x.shape)
# print(y.shape)


if __name__ == '__main__':
    test_model()
