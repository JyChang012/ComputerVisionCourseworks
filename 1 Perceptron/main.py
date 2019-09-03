import numpy as np
from matplotlib import pyplot as plt


def data_generation(num_observations=500, seed=12):
    np.random.seed(seed)

    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((-np.ones(num_observations), np.ones(num_observations)))

    return X, Y


class Perceptron:
    """A simple binary perceptron classifier."""

    def __init__(self, max_iterations=30, learning_rate=1.):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.errors = None
        self.w = None
        self.x = None
        self.y = None

    def fit(self, x=np.array([]), y=np.array([])):
        self.x = np.hstack([np.ones([x.shape[0], 1]), x])
        self.y = y
        self.w = np.zeros(self.x.shape[1])
        self.errors = []

        for i in range(self.max_iterations):
            total_error = 0
            for xi, yi in zip(self.x, self.y):
                if np.dot(self.w, xi) * yi <= 0:
                    total_error += (np.dot(self.w, xi)*yi)
                    self.w = self.w + self.learning_rate * yi * xi
            self.errors.append(-total_error)
        return self.w

    def plot(self):
        """Plot the data with separating line."""
        if self.x is None:
            raise RuntimeError('Haven\'t fitted yet!')
        elif self.x.shape[1] != 3:
            raise RuntimeError('Unsupported dimension number!')

        plt.scatter(self.x[:, 1], self.x[:, 2], c=self.y)
        x_range = np.array([self.x[:, 1].min(), self.x[:, 1].max()])
        y_range = -self.w[0] / self.w[2] - self.w[1] / self.w[2] * x_range
        plt.plot(x_range, y_range)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(f"Data with Separating Line")
        plt.show()
        print(f'total loss = {self.errors[-1]}')

    def plot_errors(self):
        """Plot errors recorded at each epoch."""
        if self.errors is None:
            raise RuntimeError('Haven\'t fitted yet!')

        plt.plot(self.errors)
        plt.xlabel('Epoch')
        plt.ylabel('Total loss')
        plt.title('Total Loss at each Epoch')
        plt.show()


if __name__ == '__main__':
    x, y = data_generation()
    classifier = Perceptron(max_iterations=30, learning_rate=.1)
    classifier.fit(x, y)
    classifier.plot()
    classifier.plot_errors()

