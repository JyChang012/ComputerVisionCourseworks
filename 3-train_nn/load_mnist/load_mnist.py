import numpy as np
from main import NN
import sys
sys.path.append('mnist')
import mnist


def load_data():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    # Flatten the data
    n_train, w, h = train_images.shape
    X_train = train_images.reshape((n_train, w*h))
    y_train = train_labels

    n_test, w, h = test_images.shape
    X_test = test_images.reshape((n_test, w*h))
    y_test = test_labels

    print(X_train.shape, y_train.shape)  # (60000, 784) (6000,)
    print(X_test.shape, y_test.shape)  # (10000, 784) (10000,)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    cls = NN(reg_lambda=.01, width=[1024, 256], activation=['tanh', 'tanh'])
    cls.fit(X_train.astype(np.float16), y_train, epoch=350, eta=.001, optimizer='Adam', verbose=True, batch_size=32)
    # epoch=240
    cls.plot_losses(save_fig=True)
    cls.score(X_test.astype(np.float16), y_test)
