import tensorflow as tf
import numpy as np
import os
import prepare_data
import models
import matplotlib.pyplot as plt
import utils
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.enable_eager_execution()

keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE


def train():
    log_name = str(datetime.datetime.now()).replace(' ', '_')[:-7]
    os.mkdir(f'./clf/logs/{log_name}')
    os.mkdir(f'./regress/logs/{log_name}')

    callbacks_clf = [
                     keras.callbacks.TensorBoard(f'./clf/logs/{log_name}'),
                     keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
    callbacks_regress = [
                         keras.callbacks.TensorBoard(f'./regress/logs/{log_name}'),
                         keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    data_classify, data_regress, n = prepare_data.get_data()
    model = models.bbox_regressor((128, 128, 3), logits_output=False, transfer=True)

    model_clf = keras.Model(inputs=model.inputs, outputs=model.outputs[0])
    model_clf.compile(optimizer=keras.optimizers.Adam(3e-6),  # 1e-5
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])
    train_data_classify, val_data_classify = prepare_data.split_data(data_classify, n)
    model_clf.fit(train_data_classify, epochs=10000, validation_data=val_data_classify, callbacks=callbacks_clf)
    # model.save('./clf/first_model.h5')

    # imgs = []
    # for label in prepare_data_v2.classes:
    #     imgs.append(plt.imread(f'./tiny_vid/{label}/000001.JPEG'))
    # imgs = np.array(imgs)
    # rsts = model_clf.predict(imgs)
    # for i, img, label in zip(range(1, imgs.shape[0] + 1), imgs, rsts):
    #     plt.subplot(3, 2, i)
    #     plt.imshow(img)
    #     label = np.argmax(label)
    #     plt.title(prepare_data_v2.classes[label])
    # plt.savefig(f'clf_full_{log_name}.jpg')
    # plt.show()
#
    # freeze conv layers
    for layer in model.layers:
        name = layer.name
        if 'block' in name:
            layer.trainable = False

    model_regress = keras.Model(inputs=model.inputs, outputs=model.outputs[1])
    model_regress.compile(optimizer=keras.optimizers.Adam(5e-7),  # too large
                          loss=keras.losses.MeanSquaredError(),
                          metrics=[keras.metrics.MeanAbsoluteError()])
    train_data_regress, val_data_regress = prepare_data.split_data(data_regress, n)
    model_regress.fit(train_data_regress, epochs=10000, validation_data=val_data_regress, callbacks=callbacks_regress)

    # Visualize

    imgs = []

    for label in prepare_data.classes:
        imgs.append(plt.imread(f'./tiny_vid/{label}/000111.JPEG'))

    imgs = np.array(imgs)

    rsts = model.predict(imgs)

    for i, img, label, cor in zip(range(1, imgs.shape[0] + 1), imgs, rsts[0], rsts[1]):
        plt.subplot(3, 2, i)
        plt.imshow(img)
        # utils.plot_box_from_xywh(cor)
        utils.draw_rectangle(cor)
        label = np.argmax(label)
        plt.xticks([]), plt.yticks([])
        plt.title(prepare_data.classes[label])

    plt.savefig(f'full_{log_name}.jpg')
    plt.show()
    pass


def train2():
    log_name = str(datetime.datetime.now()).replace(' ', '_')[:-7]
    os.mkdir(f'./clf/logs/{log_name}')
    os.mkdir(f'./regress/logs/{log_name}')

    data_classify, data_regress, n = prepare_data.get_data()
    model = models.bbox_regressor((128, 128, 3), logits_output=False, transfer=True)

    callbacks_clf = [
        keras.callbacks.TensorBoard(f'./clf/logs/{log_name}'),
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)]
    callbacks_regress = [
        keras.callbacks.TensorBoard(f'./regress/logs/{log_name}'),
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)]


    model_regress = keras.Model(inputs=model.inputs, outputs=model.outputs[1])
    model_regress.compile(optimizer=keras.optimizers.Adam(5e-7),  # too large
                          loss=keras.losses.MeanSquaredError(),
                          metrics=[keras.metrics.MeanAbsoluteError()])
    train_data_regress, val_data_regress = prepare_data.split_data(data_regress, n)
    model_regress.fit(train_data_regress, epochs=280, validation_data=val_data_regress, callbacks=callbacks_regress)
    # epoch=263
    # imgs = []
    # for label in prepare_data_v2.classes:
    #     imgs.append(plt.imread(f'./tiny_vid/{label}/000001.JPEG'))
    # imgs = np.array(imgs)
    # rsts = model_clf.predict(imgs)
    # for i, img, label in zip(range(1, imgs.shape[0] + 1), imgs, rsts):
    #     plt.subplot(3, 2, i)
    #     plt.imshow(img)
    #     label = np.argmax(label)
    #     plt.title(prepare_data_v2.classes[label])
    # plt.savefig(f'clf_full_{log_name}.jpg')
    # plt.show()
    #
    # freeze conv layers
    for layer in model.layers:
        name = layer.name
        if 'block' in name:
            layer.trainable = False

    model_clf = keras.Model(inputs=model.inputs, outputs=model.outputs[0])
    model_clf.compile(optimizer=keras.optimizers.Adam(3e-6),  # 1e-5
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])
    train_data_classify, val_data_classify = prepare_data.split_data(data_classify, n)
    model_clf.fit(train_data_classify, epochs=100, validation_data=val_data_classify, callbacks=callbacks_clf)
    # epoch=89?
    # Visualize

    class2cor = prepare_data.classes2cor

    imgs = []

    for label in prepare_data.classes:
        imgs.append(plt.imread(f'./tiny_vid/{label}/000032.JPEG'))

    imgs = np.array(imgs)

    rsts = model.predict(keras.applications.vgg16.preprocess_input(imgs))

    for i, img, label, cor in zip(range(1, imgs.shape[0] + 1), imgs, rsts[0], rsts[1]):
        plt.subplot(3, 2, i)
        plt.imshow(img)
        utils.plot_box_from_xywh(cor)
        # utils.plot_box_from_xywh(class2cor[prepare_data_v2.classes[label]][i-1])
        label = np.argmax(label)
        plt.xticks([]), plt.yticks([])
        plt.title(prepare_data.classes[label])

    model.save(f'./model_{log_name}.h5')

    plt.savefig(f'full_{log_name}.jpg')
    plt.show()
    pass


if __name__ == '__main__':
    train2()
