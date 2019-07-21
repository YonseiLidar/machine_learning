import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


class LinearRegression:

    def __init__(self, x_train, y_train, sess, lr=1e-3, epoch=1000, batch_size=10, np_seed=1, tf_seed=1,
                 initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
                 save_path='linear_regression/model/lr_1'):
        self.x_train = x_train
        self.y_train = y_train
        self.sess = sess
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.np_seed = np_seed
        self.tf_seed = tf_seed
        self.initializer = initializer
        self.save_path = save_path

        self.save_dir = self.save_path.split('/')[0]
        self.feature_dim = self.x_train.shape[1]
        self.coeff = None

        self.x, self.y, self.y_pred = None, None, None
        self.W, self.b = None, None
        self.cost, self.opt = None, None
        self.saver = None

    def compute_gradient(self, x, y, theta):
        a = self.generate_design_matrix(x)
        return 2 * a.T @ a @ theta - 2 * a.T @ y

    def construct_graph(self):
        np.random.seed(self.np_seed)
        tf.set_random_seed(self.tf_seed)

        input_size = self.feature_dim

        self.x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

        self.W, self.b = self.layer_w_and_b([input_size, 1], [1], 'W', 'b')
        self.y_pred = tf.identity(tf.matmul(self.x, self.W) + self.b, name='y_pred')
        self.cost = tf.nn.l2_loss(self.y - self.y_pred)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def fit_measure(self, y, y_pred):
        print("mean squared error : ", self.mean_squared_error(y, y_pred))
        print("R square           : ", self.r2_score(y, y_pred))

    @staticmethod
    def generate_design_matrix(x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    def layer_w_and_b(self, w_shape, b_shape, w_name, b_name):
        w = tf.get_variable(w_name, shape=w_shape, initializer=self.initializer)
        b = tf.get_variable(b_name, initializer=tf.zeros(b_shape))
        return w, b

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
        feed_dict = {self.x: x}
        return self.sess.run(self.y_pred, feed_dict=feed_dict)

    def restore(self):
        self.saver = tf.train.import_meta_graph(self.save_path + '.meta', clear_devices=True)
        self.saver.restore(sess=self.sess, save_path=self.save_path)

        self.x = self.sess.graph.get_tensor_by_name('x:0')
        self.y_pred = self.sess.graph.get_tensor_by_name('y_pred:0')

    @staticmethod
    def r2_score(y_train, y_train_pred):
        upper = np.sum((y_train - y_train_pred)**2)
        lower = np.sum((y_train - y_train.mean())**2)
        return 1 - (upper / lower)

    def save(self):
        self.saver = tf.train.Saver()
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.saver.save(sess=self.sess, save_path=self.save_path)

    @staticmethod
    def shuffle_data(*args):
        idx = np.arange(args[0].shape[0])
        np.random.shuffle(idx)
        list_to_return = []
        for arg in args:
            list_to_return.append(arg[idx])
        return list_to_return

    def train(self):
        self.construct_graph()
        tf.global_variables_initializer().run()

        for _ in range(self.epoch):
            x, y = self.shuffle_data(self.x_train, self.y_train)
            for i in range(self.x_train.shape[0] // self.batch_size):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.x: x_batch, self.y: y_batch}
                self.sess.run(self.opt, feed_dict=feed_dict)

        b, w = self.sess.run([self.b, self.W])
        self.coeff = np.concatenate((b.reshape((-1, 1)), w), axis=0)

        self.save()
