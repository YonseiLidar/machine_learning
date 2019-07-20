import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, x_train, y_train, lr=1e-3, epoch=1000, batch_size=10, np_seed=1, tf_seed=1):
        self.x_train = x_train
        self.y_train = y_train
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.np_seed = np_seed
        self.tf_seed = tf_seed

        self.feature_dim = self.x_train.shape[1]
        self.coeff = None

    def compute_gradient(self, x, y, theta):
        a = self.generate_design_matrix(x)
        return (2 / self.batch_size) * (a.T @ a @ theta - a.T @ y)

    def fit_measure(self, y, y_pred):
        print("mean squared error : ", self.mean_squared_error(y, y_pred))
        print("R square           : ", self.r2_score(y, y_pred))

    @staticmethod
    def generate_design_matrix(x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    @staticmethod
    def mean_squared_error(y_train, y_train_pred):
        upper = np.sum((y_train - y_train_pred)**2)
        lower = y_train.shape[0]
        return upper / lower

    @staticmethod
    def plot_y_and_y_pred(x, y, y_pred):
        plt.plot(x, y, 'o')
        plt.plot(x, y_pred)
        plt.show()

    def predict(self, x):
        a = self.generate_design_matrix(x)
        return a @ self.coeff

    @staticmethod
    def r2_score(y_train, y_train_pred):
        upper = np.sum((y_train - y_train_pred)**2)
        lower = np.sum((y_train - y_train.mean())**2)
        return 1 - (upper / lower)

    @staticmethod
    def shuffle_data(*args):
        idx = np.arange(args[0].shape[0])
        np.random.shuffle(idx)
        list_to_return = []
        for arg in args:
            list_to_return.append(arg[idx])
        return list_to_return

    def train(self):
        np.random.seed(self.np_seed)

        self.coeff = np.random.normal(0., 1., (self.x_train.shape[1] + 1, 1))
        for _ in range(self.epoch):
            x, y = self.shuffle_data(self.x_train, self.y_train)
            for i in range(self.x_train.shape[0] // self.batch_size):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
                grad = self.compute_gradient(x_batch, y_batch, self.coeff)
                self.coeff -= self.lr * grad
