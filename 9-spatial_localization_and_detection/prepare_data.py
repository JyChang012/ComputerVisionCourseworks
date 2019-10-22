import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Get classes list
classes = ['bird', 'car', 'dog', 'lizard', 'turtle']
idices = tf.convert_to_tensor([i for i in range(5)])
classes2idx = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(classes, idices, key_dtype=tf.string,
                                                                            value_dtype=tf.int32), default_value=-1,
                                        name='idx_lookup')


# Get coordinates
# coordinates = np.array([np.loadtxt(f'./tiny_vid/{c}_gt.txt', dtype=np.int)[:, 1:] for c in classes])


classes2cor = []
for c in classes:
    cor = np.loadtxt(f'./tiny_vid/{c}_gt.txt', dtype=np.int)
    cor = cor[:180, 1:]
    cor = np.hstack((np.mean(cor[:, (0, 2)], axis=1, dtype=np.int, keepdims=True),
                     np.mean(cor[:, (1, 3)], axis=1, dtype=np.int, keepdims=True),
                     (cor[:, 2] - cor[:, 0]).reshape(-1, 1),
                     (cor[:, 3] - cor[:, 1]).reshape(-1, 1)))
    classes2cor.append(tf.convert_to_tensor(cor))

# classes2cor = np.array(classes2cor)
# classes2cor = tf.convert_to_tensor(classes2cor)

classes2cor = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
    keys=classes, values=classes2cor, key_dtype=tf.string, value_dtype=tf.int64
), name='find_cor', default_value=False)

pass


# draw the first img
def draw_rectangle(bounding):  # bounding = [xmin, ymin, xmax, ymax]
    x = [bounding[0], bounding[0], bounding[2], bounding[2]]
    y = [bounding[1], bounding[3], bounding[3], bounding[1]]
    plt.plot(x, y)


def test_data():
    img = cv.imread('./tiny_vid/bird/000001.JPEG')
    bird_annote = np.loadtxt('./tiny_vid/bird_gt.txt', dtype=np.int)
    # each line is (image_index, xmin, ymin, xmax, ymax), where x is horizontal coordinate and y is vertical with origin
    # at upper left
    plt.imshow(img)
    draw_rectangle(bird_annote[0, 1:])
    plt.show()


# preprocess data
def process_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def process_img(img):
    img = tf.cast(img, dtype=tf.float32)
    img = keras.applications.vgg16.preprocess_input(img)
    return img


def preprocess(file_path):
    parts = tf.strings.split([file_path], '/', result_type='RaggedTensor')
    label = parts[-2]
    name = tf.cast(parts[-1][:-5], tf.int32) - 1
    img = process_path(file_path)
    img = process_img(img)
    return img, classes2idx[label], classes2cor.lookup(label)[name]


def get_data():
    data = tf.data.Dataset.list_files('*.JPEG')
    data = data.map(preprocess, num_parallel_calls=AUTOTUNE)
    data_cls = data.map(lambda img, idx, cor: (img, idx), num_parallel_calls=AUTOTUNE)
    data_cor = data.map(lambda img, idx, cor: (img, cor), num_parallel_calls=AUTOTUNE)
    pass


if __name__ == '__main__':
    get_data()
    pass
