import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('mnist')
import mnist

keras = tf.keras


# tf.enable_eager_execution()


def load_data():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    return train_images, test_images, train_labels, test_labels


def preprocess():
    train_images, test_images, train_labels, test_labels = load_data()

    train_data = tf.data.Dataset.from_tensor_slices((
        train_images[..., tf.newaxis] / 255,
        train_labels.astype(np.int16)))

    test_data = tf.data.Dataset.from_tensor_slices((
        test_images[..., tf.newaxis] / 255,
        test_labels.astype(np.int16)))

    train_data = train_data.shuffle(1000).batch(32)
    test_data = test_data.shuffle(1000).batch(32)

    return train_data, test_data


def train1():
    train_data, test_data = preprocess()

    callbacks = [keras.callbacks.TensorBoard()]

    model = keras.Sequential([keras.layers.Conv2D(16, [3, 3], activation='relu'),
                              keras.layers.Conv2D(16, [3, 3], activation='relu'),
                              keras.layers.Conv2D(8, [3, 3], activation='relu'),
                              keras.layers.Conv2D(4, [3, 3], activation='relu'),
                              keras.layers.MaxPool2D(),
                              keras.layers.Flatten(),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(train_data, epochs=128, steps_per_epoch=14, validation_data=test_data, callbacks=callbacks)

    model.save('./models/first_model')

    # for img, lab in test_data.take(1):
    #     logit = model(img[0:1])
    #     loss = tf.losses.sparse_softmax_cross_entropy(lab[0:1], logit)
    # pass


def train2():
    train_data, test_data = preprocess()
    callbacks = [keras.callbacks.TensorBoard(), keras.callbacks.EarlyStopping(patience=15)]

    model = keras.Sequential([keras.layers.Conv2D(64, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(64, [3, 3], activation='relu', padding='same'),
                              keras.layers.MaxPool2D(padding='same'),
                              keras.layers.Conv2D(128, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(128, [3, 3], activation='relu', padding='same'),
                              keras.layers.MaxPool2D(padding='same'),
                              keras.layers.Conv2D(256, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(256, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(256, [3, 3], activation='relu', padding='same'),
                              keras.layers.MaxPool2D(padding='same'),
                              keras.layers.Conv2D(512, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(512, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(512, [3, 3], activation='relu', padding='same'),
                              keras.layers.MaxPool2D(padding='same'),
                              keras.layers.Conv2D(512, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(512, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(512, [3, 3], activation='relu', padding='same'),
                              keras.layers.MaxPool2D(padding='same'),
                              keras.layers.Flatten(),
                              keras.layers.Dense(4096, activation='relu'),
                              keras.layers.Dense(4096, activation='relu'),
                              keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(train_data, epochs=130, steps_per_epoch=14, validation_data=test_data, callbacks=callbacks)
    model.save('./models/vgg')


def train3():
    train_data, test_data = preprocess()

    callbacks = [keras.callbacks.TensorBoard(log_dir='./logs2')]

    model = keras.Sequential([keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same'),
                              keras.layers.MaxPool2D(padding='same'),
                              keras.layers.Conv2D(8, [3, 3], activation='relu', padding='same'),
                              keras.layers.Conv2D(4, [3, 3], activation='relu', padding='same'),
                              keras.layers.MaxPool2D(padding='same'),
                              keras.layers.Flatten(),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(train_data, epochs=128, steps_per_epoch=14, validation_data=test_data, callbacks=callbacks)

    model.save('./models/3rg_model')

    # for img, lab in test_data.take(1):
    #     logit = model(img[0:1])
    #     loss = tf.losses.sparse_softmax_cross_entropy(lab[0:1], logit)
    # pass

    model.evaluate()


def examine_false():
    tf.enable_eager_execution()
    train_data, test_data = preprocess()
    model = tf.keras.models.load_model('./models/3rg_model')
    est = model.predict(test_data)
    est = np.argmax(est, axis=1)
    # for pred, (img, label) in zip(est, list(test_data.enumerate())):
    #     if pred != label:
    #         pass
    test_data = test_data.apply(tf.data.experimental.unbatch())
    cnt = 1
    for pred, (img, label) in zip(est, test_data):
        if pred != label.numpy() and cnt < 9:
            plt.imshow(img.numpy().reshape([28, 28]), cmap='Greys_r')
            plt.title(f'predict = {pred}, truth = {label}')
            plt.savefig(f'{cnt}.svg')
            cnt += 1

    # plt.savefig('false.svg')


if __name__ == '__main__':
    examine_false()
