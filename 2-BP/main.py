import numpy as np
from matplotlib import pyplot as plt
import sklearn.datasets as datasets
from scipy import special
logsumexp = special.logsumexp
softmax = special.softmax


def relu(Z):
    return Z * (Z > 0)


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y)


def calculate_err(y, y_hat):
    err = np.sum(y != y_hat) / y.shape[0]
    return err


class NN:
    """A simple 2 layers neural network classifier."""

    def __init__(self, iter_num=10000, eta=.01, reg_lambda=.01, width=8, activation='tanh'):
        self.iter_num = iter_num
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.width = width
        self.activation = activation
        self.X = None
        self.y = None
        self.weights = dict(W1=None, W2=None, b1=None, b2=None)
        self.nodes = dict(Z1=None, A1=None, Z2=None, O=None)
        self.losses = None
        self.error_rate = None

    def forward_pass(self, X):
        self.nodes['Z1'] = X @ self.weights['W1'].T + self.weights['b1']
        if self.activation == 'tanh':
            self.nodes['A1'] = np.tanh(self.nodes['Z1'])
        elif self.activation == 'relu':
            self.nodes['A1'] = relu(self.nodes['Z1'])
        else:
            raise ValueError('Unsupported')
        self.nodes['Z2'] = self.nodes['A1'] @ self.weights['W2'].T + self.weights['b2']
        # exp = np.exp(self.nodes['Z2'])
        # self.nodes['O'] = exp / np.sum(exp, axis=1, keepdims=True)
        # self.nodes['O'] = np.exp(self.nodes['Z2'] - logsumexp(self.nodes['Z2'], axis=1, keepdims=True))
        self.nodes['O'] = softmax(self.nodes['Z2'], axis=1)

    def predict(self, X):
        self.forward_pass(X)
        label = np.argmax(self.nodes['O'], axis=1)
        return label

    def score(self, X, y):
        y_hat = self.predict(X)
        err = calculate_err(y, y_hat)
        print(f'err = {err}')
        return err

    def backward_pass(self, X, y):

        # Calculate gradients
        dZ2 = self.nodes['O'].copy()
        dZ2[range(dZ2.shape[0]), y] -= 1
        dW2 = dZ2.T @ self.nodes['A1']
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.weights['W2']
        if self.activation == 'relu':
            dZ1 = (dA1 > 0) * 1
        elif self.activation == 'tanh':
            dZ1 = dA1 * (1 - self.nodes['A1'] ** 2)
        dW1 = dZ1.T @ X
        db1 = np.sum(dZ1, axis=0)
        # Add regularization term
        dW1 += self.reg_lambda * self.weights['W1']
        dW2 += self.reg_lambda * self.weights['W2']
        # Update weights
        self.weights['W1'] -= (self.eta * dW1)
        self.weights['W2'] -= (self.eta * dW2)
        self.weights['b1'] -= (self.eta * db1)
        self.weights['b2'] -= (self.eta * db2)

    def loss(self, batch_size=None):
        if batch_size is not None:
            self.forward_pass(self.X)
        j = np.sum(logsumexp(self.nodes['Z2'], axis=1) - self.nodes['Z2'][range(self.X.shape[0]), self.y], axis=0) / \
            self.X.shape[0]
        return j

    def fit(self, X, y, verbose=True, batch_size=None):
        self.X = X
        self.y = y
        class_num = np.max(y) + 1
        self.weights = dict(W1=np.random.normal(0, 1e-2, [self.width, self.X.shape[1]]),
                            W2=np.random.normal(0, 1e-2, [class_num, self.width]),
                            b1=np.random.normal(0, 0, self.width),
                            b2=np.random.normal(0, 0, class_num))
        self.losses = []

        for i in range(self.iter_num):
            if batch_size is None:
                self.forward_pass(self.X)
                self.backward_pass(self.X, self.y)
            else:
                choices = np.random.choice(self.X.shape[0], size=batch_size)
                self.forward_pass(self.X[choices, :])
                self.backward_pass(self.X[choices, :], self.y[choices])
            j = self.loss(batch_size=batch_size)
            self.losses.append(j)
            if verbose is True:
                print(f'Epoch {i}, loss = {j}')

        y_hat = self.predict(self.X)
        self.error_rate = calculate_err(self.y, y_hat)
        print(f'err = {self.error_rate}')

    def plot_boundary(self, save_fig=False, file_name='Data with Decision Boundary.svg'):
        if self.X is None:
            raise RuntimeError('Have not fitted yet!')
        elif self.X.shape[1] != 2:
            raise ValueError('Unsupported dimension!')
        else:
            plot_decision_boundary(lambda X: self.predict(X), self.X, self.y)
            plt.title(f'Data with Decision Boundary: err = {self.error_rate}')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            if save_fig is True:
                plt.savefig(file_name)
            plt.show()

    def plot_losses(self, save_fig=False, file_name='Loss at Each Epoch.svg'):
        plt.plot(self.losses)
        plt.title(f'Loss at Each Epoch: err = {self.error_rate}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if save_fig is True:
            plt.savefig(file_name)
        plt.show()


def task1():
    X, y = datasets.make_moons(200, noise=0.20)
    cls = NN(iter_num=3000, eta=.1, reg_lambda=.005, width=16)
    cls.fit(X, y, verbose=False, batch_size=32)
    cls.plot_losses(save_fig=False)
    cls.plot_boundary(save_fig=False)


if __name__ == '__main__':
    task1()
