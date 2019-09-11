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

    def __init__(self, reg_lambda=.01, width=[8, 8], activation=['tanh', 'tanh']):

        self.reg_lambda = reg_lambda
        self.width = width
        self.activation = activation
        self.X = None
        self.y = None
        self.weights = dict(W1=None, W2=None, W3=None, b1=None, b2=None, b3=None)
        self.gradients = dict.fromkeys(self.weights)
        self.nodes = dict()
        self.losses = None
        self.error_rate = None

    def forward_pass(self, X):

        self.nodes['Z1'] = X @ self.weights['W1'].T + self.weights['b1']  # Has this executed?
        if self.activation[0] == 'tanh':
            self.nodes['A1'] = np.tanh(self.nodes['Z1'])
        elif self.activation[0] == 'relu':
            self.nodes['A1'] = relu(self.nodes['Z1'])
        else:
            raise ValueError('Unsupported')
        self.nodes['Z2'] = self.nodes['A1'] @ self.weights['W2'].T + self.weights['b2']
        if self.activation[1] == 'tanh':
            self.nodes['A2'] = np.tanh(self.nodes['Z2'])
        elif self.activation[1] == 'relu':
            self.nodes['A2'] = relu(self.nodes['Z2'])
        self.nodes['Z3'] = self.nodes['A2'] @ self.weights['W3'].T + self.weights['b3']
        # exp = np.exp(self.nodes['Z2'])
        # self.nodes['O'] = exp / np.sum(exp, axis=1, keepdims=True)
        # self.nodes['O'] = np.exp(self.nodes['Z2'] - logsumexp(self.nodes['Z2'], axis=1, keepdims=True))
        self.nodes['O'] = softmax(self.nodes['Z3'], axis=1)

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
        dZ3 = self.nodes['O'].copy()
        dZ3[list(range(dZ3.shape[0])), y] -= 1
        dZ3 /= y.shape[0]
        self.gradients['W3'] = dZ3.T @ self.nodes['A2']
        self.gradients['b3'] = np.sum(dZ3, axis=0)
        dA2 = dZ3 @ self.weights['W3']
        if self.activation[0] == 'relu':
            dZ2 = (dA2 > 0) * 1
        elif self.activation[0] == 'tanh':
            dZ2 = dA2 * (1 - self.nodes['A2'] ** 2)
        self.gradients['W2'] = dZ2.T @ self.nodes['A1']
        self.gradients['b2'] = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.weights['W2']
        if self.activation[1] == 'relu':
            dZ1 = (dA1 > 0) * 1
        elif self.activation[1] == 'tanh':
            dZ1 = dA1 * (1 - self.nodes['A1'] ** 2)
        self.gradients['W1'] = dZ1.T @ X
        self.gradients['b1'] = np.sum(dZ1, axis=0)
        # Add regularization term
        for key in self.gradients.keys():
            if key[0] == 'W':
                self.gradients[key] += self.reg_lambda * self.weights[key]

    def sgd_update(self, eta=.01):
        for key in self.weights:
            self.weights[key] -= (eta * self.gradients[key])

    def adam_update(self, s, r, t=0, eta=.001, p1=0.9, p2=0.999):
        for key in self.gradients:
            s[key] = p1 * s[key] + (1 - p1) * self.gradients[key]
            r[key] = p2 * r[key] + (1 - p2) * (self.gradients[key] ** 2)
            s_hat = s[key] / (1 - p1 ** t)
            r_hat = r[key] / (1 - p2 ** t)
            delta = -eta * s_hat / (np.sqrt(r_hat) + 1e-8)
            self.weights[key] += delta

    def loss(self, batch_size=None):
        if batch_size is not None:
            self.forward_pass(self.X)
        j = np.sum(logsumexp(self.nodes['Z3'], axis=1) - self.nodes['Z3'][range(self.X.shape[0]), self.y], axis=0) / \
            self.X.shape[0]
        return j

    def fit(self, X, y, epoch=10000, eta=.001, optimizer='SGD', verbose=True, batch_size=None, p1=0.9, p2=0.999):
        self.X = X
        self.y = y
        # Initialize 1st and 2nd momentum
        class_num = np.max(y) + 1
        # self.weights = dict(W1=np.random.normal(0, 1e-7, [self.width[0], self.X.shape[1]]),
        #                     W2=np.random.normal(0, 1e-7, [self.width[1], self.width[0]]),
        #                     W3=np.random.normal(0, 1e-7, [class_num, self.width[1]]),
        #                     b1=np.random.normal(0, 1e-3, self.width[0]),
        #                     b2=np.random.normal(0, 1e-3, self.width[1]),
        #                     b3=np.random.normal(0, 1e-3, class_num))

        self.weights = dict(W1=np.random.randn(self.width[0], self.X.shape[1]) / np.sqrt(self.X.shape[1] / 2),
                            W2=np.random.randn(self.width[1], self.width[0]) / np.sqrt(self.width[0] / 2),
                            W3=np.random.randn(class_num, self.width[1]) / np.sqrt(self.width[1] / 2),
                            b1=np.random.normal(0, 1e-3, self.width[0]),
                            b2=np.random.normal(0, 1e-3, self.width[1]),
                            b3=np.random.normal(0, 1e-3, class_num))

        self.losses = []
        if optimizer == 'Adam':
            s = dict()
            r = dict()
            for key in self.weights:
                s[key] = np.zeros(self.weights[key].shape)
                r[key] = s[key].copy()

        for t in range(epoch):
            if batch_size is None:
                self.forward_pass(X)
                self.backward_pass(X, y)
            else:
                choices = np.random.choice(self.X.shape[0], size=batch_size, replace=True)
                choices_x = X[choices, :]
                choices_y = y[choices]
                self.forward_pass(X=choices_x)
                self.backward_pass(X=choices_x, y=choices_y)
            if optimizer == 'SGD':
                self.sgd_update(eta)
            elif optimizer == 'Adam':
                self.adam_update(s, r, t+1, eta, p1, p2)

            j = self.loss(batch_size=batch_size)
            self.losses.append(j)
            if verbose is True:
                print(f'Epoch {t}, loss = {j}')

        y_hat = self.predict(self.X)
        self.error_rate = calculate_err(self.y, y_hat)
        print(f'err = {self.error_rate}')

    def plot_boundary(self, save_fig=False, file_name='Data_with_Decision_Boundary.svg'):
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

    def plot_losses(self, save_fig=False, file_name='Loss_at_Each_Epoch.svg'):
        plt.plot(self.losses)
        plt.title(f'Loss at Each Epoch: err = {self.error_rate}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if save_fig is True:
            plt.savefig(file_name)
        plt.show()


def task1():
    X, y = datasets.make_moons(200, noise=0.20)
    cls = NN(reg_lambda=.005, width=[16, 8], activation=['tanh', 'tanh'])
    cls.fit(X, y, verbose=False, batch_size=32, optimizer='Adam', eta=.06, epoch=20000)
    cls.plot_losses(save_fig=True)
    cls.plot_boundary(save_fig=True)


if __name__ == '__main__':
    task1()
