import os

import numpy as np
import tensorflow as tf

import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# tf.enable_eager_execution()

keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE

classes = ['bird', 'car', 'dog', 'lizard', 'turtle']


def data_map(path, x, y, w, h, label):
    # path, x, y, w, h, label = row
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.cast(img, dtype=tf.float32)
    img = keras.applications.vgg16.preprocess_input(img)

    coordinate = tf.cast(tf.convert_to_tensor([x, y, w, h]), tf.float32)
    label = tf.cast(label, tf.int64)

    return img, coordinate, label


def split_data(data, n, split=0.8, batch_size=32):
    train_data = data.take(int(split * n))
    val_data = data.skip(int(split * n))

    if batch_size:
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size)

    return train_data, val_data


classes2cor = dict()

for c in classes:
    cor = np.loadtxt(f'./tiny_vid/{c}_gt.txt', dtype=np.int)
    cor = cor[:180, 1:].astype(np.float32)
    cor = utils.bounding_xywh_transform(cor, b2xywh=True, batch=True)
    classes2cor[c] = cor


def get_data():
    class2idx = {cls: i for i, cls in enumerate(classes)}

    paths = []
    xs = []
    ys = []
    ws = []
    hs = []
    labels = []
    for ci, c in enumerate(classes):
        for name in range(180):
            paths.append(f'./tiny_vid/{c}/{name+1:0>6}.JPEG'.encode('utf-8'))
            xs.append(classes2cor[c][name, 0])
            ys.append(classes2cor[c][name, 1])
            ws.append(classes2cor[c][name, 2])
            hs.append(classes2cor[c][name, 3])
            labels.append(ci)

    paths = np.array(paths)

    data = tf.data.Dataset.from_tensor_slices((paths, xs, ys, ws, hs, labels))
    data = data.shuffle(1000)
    # string should be converted as binary (encoded with utf-8)
    data = data.map(data_map, num_parallel_calls=AUTOTUNE)

    data_classify = data.map(lambda img, cor, label: (img, label), num_parallel_calls=AUTOTUNE)
    data_regress = data.map(lambda img, cor, label: (img, cor), num_parallel_calls=AUTOTUNE)

    return data_classify, data_regress, len(xs)


if __name__ == '__main__':
    get_data()
