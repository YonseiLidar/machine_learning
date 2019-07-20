import numpy as np
import scipy.linalg as la
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
        return 2 * a.T @ a @ theta - 2 * a.T @ y

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

    def train(self):
        a = self.generate_design_matrix(self.x_train)
        y = self.y_train
        self.coeff = la.inv(a.T @ a) @ (a.T @ y)
