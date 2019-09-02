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

    def __init__(self, max_iterations=30, learning_rate=1.):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.errors = []
        self.w = None
        self.x = None
        self.y = None

    def fit(self, x=np.array([]), y=np.array([])):
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        w = np.zeros(x.shape[1])

        for i in range(self.max_iterations):
            total_error = 0
            for xi, yi in zip(x, y):
                if np.dot(w, xi) * yi <= 0:
                    total_error += (np.dot(w, xi)*yi)
                    w = w + self.learning_rate * yi * xi
            self.errors.append(-total_error)
        self.w = w
        self.x = x
        self.y = y
        return self.w

    def plot(self):
        if self.x is None:
            print('Haven\'t fitted yet!!')
            return
        elif self.x.shape[1] != 3:
            print('Unsupported dimension number!')
            return

        plt.scatter(self.x[:, 1], self.x[:, 2], c=self.y)
        x_range = np.array([self.x[:, 1].min(), self.x[:, 1].max()])
        y_range = -self.w[0] / self.w[2] - self.w[1] / self.w[2] * x_range
        plt.plot(x_range, y_range)
        plt.show()

    def plot_errors(self):
        if self.x is None:
            print('Haven\'t fitted yet!!')
            return

        plt.plot(self.errors)
        plt.xlabel('Epoch')
        plt.ylabel('Total loss')
        plt.title('Total Loss at each Epoch')
        plt.show()


if __name__ == '__main__':
    x, y = data_generation()
    classifier = Perceptron(max_iterations=300, learning_rate=0.01)
    classifier.fit(x, y)
    classifier.plot()
    classifier.plot_errors()

