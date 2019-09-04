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

    def __init__(self, max_iterations=30, learning_rate=1., shuffle=True):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.shuffled = shuffle
        self.losses = None
        self.w = None
        self.x = None
        self.y = None
        self.error_rate = None

    def fit(self, x=np.array([]), y=np.array([])):
        self.x = np.hstack([np.ones([x.shape[0], 1]), x])
        self.y = y
        self.w = np.random.normal(0, 1e-3, self.x.shape[1])
        # Quote: Now, the reason we don't initialize the weights to zero is that the learning rate (eta) only has an
        # effect on the classification outcome if the weights are initialized to non-zero values. If all the weights are
        # initialized to zero, the learning rate parameter eta affects only the scale of the weight vector, not the
        # direction.
        self.losses = []

        for i in range(self.max_iterations):
            total_loss = 0
            if self.shuffled is True:
                shuffled = np.hstack([self.x, self.y.reshape(-1, 1)])
                np.random.shuffle(shuffled)
                _x = shuffled[:, :-1]
                _y = shuffled[:, -1]
            else:
                _x, _y = self.x, self.y
            for xi, yi in zip(_x, _y):
                if np.dot(self.w, xi) * yi <= 0:
                    total_loss += (np.dot(self.w, xi)*yi)
                    self.w = self.w + self.learning_rate * yi * xi
            self.losses.append(-total_loss)

            self.error_rate = 0
            for xi, yi in zip(self.x, self.y):
                if np.dot(self.w, xi) * yi <= 0:
                    self.error_rate += 1
            self.error_rate /= self.x.shape[0]
        return self.w

    def plot(self, save_file=False, file_name='Data_with_Separating_Line.svg'):
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
        plt.title(f"Data with Separating Line: error_rate = {self.error_rate}")
        if save_file is False:
            plt.show()
        else:
            plt.savefig(file_name)
            plt.show()

    def plot_losses(self, save_file=False, file_name='Total_Loss_at_each_Epoch.svg'):
        """Plot losses recorded at each epoch."""
        if self.losses is None:
            raise RuntimeError('Haven\'t fitted yet!')

        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Total loss')
        plt.title(f'Total Loss at each Epoch: error_rate = {self.error_rate}')
        if save_file is False:
            plt.show()
        else:
            plt.savefig(file_name)
            plt.show()


if __name__ == '__main__':
    x, y = data_generation()
    classifier = Perceptron(max_iterations=30, learning_rate=.1, shuffle=True)
    classifier.fit(x, y)
    classifier.plot(save_file=False)
    classifier.plot_losses(save_file=False)

