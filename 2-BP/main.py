import numpy as np
from matplotlib import pyplot as plt
import sklearn.datasets as datasets


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


class NN:
    """A simple 2 layers neural network binary classifer."""

    def __init__(self, iter_num=10000, eta=.01, reg_lambda=.01, width=8):
        self.iter_num = iter_num
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.width = width
        self.X = None
        self.y = None
        self.weights = dict(W1=None, W2=None, b1=None, b2=None)
        self.nodes = dict(Z1=None, A1=None, Z2=None, O=None)
        self.losses = None

    def forward_pass(self, X):
        self.nodes['Z1'] = X @ self.weights['W1'].T + self.weights['b1']
        self.nodes['A1'] = np.tanh(self.nodes['Z1'])
        self.nodes['Z2'] = self.nodes['A1'] @ self.weights['W2'].T + self.weights['b2']
        exp = np.exp(self.nodes['Z2'])
        self.nodes['O'] = exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X):
        self.forward_pass(X)
        label = np.argmax(self.nodes['O'], axis=1)
        return label

    def backward_pass(self):

        # Calculate gradients
        dZ2 = self.nodes['O'].copy()
        dZ2[range(dZ2.shape[0]), self.y] -= 1
        dW2 = dZ2.T @ self.nodes['A1']
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.weights['W2']
        dZ1 = dA1 * (1 - self.nodes['A1'] ** 2)
        dW1 = dZ1.T @ self.X
        db1 = np.sum(dZ1, axis=0)
        # Add regularization term
        dW1 += self.reg_lambda * self.weights['W1']
        dW2 += self.reg_lambda * self.weights['W2']
        # Update weights
        self.weights['W1'] -= (self.eta * dW1)
        self.weights['W2'] -= (self.eta * dW2)
        self.weights['b1'] -= (self.eta * db1)
        self.weights['b2'] -= (self.eta * db2)

    def loss(self):
        j = -np.sum(np.log(self.nodes['O'][range(self.nodes['O'].shape[0]), self.y])) / self.y.shape[0]
        return j

    def fit(self, X, y, verbose=True):
        self.X = X
        self.y = y
        class_num = np.max(y) + 1
        self.weights = dict(W1=np.random.normal(0, 1e-3, [self.width, self.X.shape[1]]),
                            W2=np.random.normal(0, 1e-3, [class_num, self.width]), b1=np.zeros(self.width),
                            b2=np.zeros(class_num))
        self.losses = []

        for i in range(self.iter_num):
            self.forward_pass(self.X)
            j = self.loss()
            self.losses.append(j)
            if verbose is True:
                print(f'Epoch {i}, loss = {j}')
            self.backward_pass()

    def plot_boundary(self, save_fig=False, file_name='Data with Decision Boundary.svg'):
        if self.X is None:
            raise RuntimeError('Have not fitted yet!')
        elif self.X.shape[1] != 2:
            raise ValueError('Unsupported dimension!')
        else:
            plot_decision_boundary(lambda X: self.predict(X), self.X, self.y)
            plt.title(f'Data with Decision Boundary')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            if save_fig is True:
                plt.savefig(file_name)
            plt.show()

    def plot_losses(self, save_fig=False, file_name='Loss at Each Epoch.svg'):
        plt.plot(self.losses)
        plt.title(f'Loss at Each Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if save_fig is True:
            plt.savefig(file_name)
        plt.show()


if __name__ == '__main__':
    X, y = datasets.make_moons(200, noise=0.20)
    cls = NN(iter_num=1000, eta=.01, reg_lambda=.01, width=12)
    cls.fit(X, y, verbose=False)
    cls.plot_losses(save_fig=True)
    cls.plot_boundary(save_fig=True)
