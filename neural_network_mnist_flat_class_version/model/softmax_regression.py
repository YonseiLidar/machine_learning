import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

from utils.utils import plot_many_images_2d


class SoftmaxRegression:

    def __init__(self, x_train, y_train, y_train_cls, cls_names, sess,
                 lr=1e-3, epoch=15, batch_size=100, report_period=1000, np_seed=1, tf_seed=1,
                 drop_out_rate=0.3,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 save_path='result/model/model_1',
                 figure_save_dir='result/img',
                 activation=tf.nn.relu):
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_cls = y_train_cls
        self.cls_names = cls_names
        self.sess = sess
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.report_period = report_period
        self.np_seed = np_seed
        self.tf_seed = tf_seed
        self.drop_out_rate = drop_out_rate
        self.initializer = initializer
        self.save_path = save_path
        self.activation = activation

        self.save_dir = self.save_path.split('/')[0]
        self.feature_dim = self.x_train.shape[1]
        self.coeff = None
        self.feature_dim = self.x_train.shape[1]
        n = int(np.sqrt(self.feature_dim))
        self.img_shape = (n, n)
        self.save_dir = self.save_path.split('/')[0]

        self.x, self.y, self.y_cls, self.y_pred = None, None, None, None
        self.W, self.b = None, None
        self.cost, self.opt = None, None
        self.logits, self.y_pred_cls = None, None
        self.entropy = None
        self.correct_bool, self.accuracy = None, None
        self.saver = None

    # def compute_accuracy(self, x, y, y_cls):
    #     feed_dict = {self.x: x, self.y: y, self.y_cls: y_cls.reshape((-1, 1))}
    #     return self.sess.run(self.accuracy, feed_dict=feed_dict)

    def compute_accuracy(self, x, y, y_cls):
        num_total = y.shape[0]
        num_correct = 0
        i = 0
        cls_pred = np.zeros(shape=num_total, dtype=np.int)
        while i < num_total:
            j = min(i + self.batch_size, num_total)
            if j == num_total:
                x_batch = x[i:]
                _, temp = self.predict(x_batch)
                cls_pred[i:] = temp.reshape((-1,))
                num_correct += np.sum(cls_pred[i:] == y_cls[i:])
            else:
                x_batch = x[i:j]
                _, temp = self.predict(x_batch)
                cls_pred[i:j] = temp.reshape((-1,))
                num_correct += np.sum(cls_pred[i:j] == y_cls[i:j])
            i = j
        return num_correct / num_total

    def compute_confusion_matrix(self, x, y, y_cls):
        num = y.shape[0]
        i = 0
        cls_pred = np.zeros(shape=num, dtype=np.int)
        while i < num:
            j = min(i + self.batch_size, num)
            if j == num:
                x_batch = x[i:]
                _, temp = self.predict(x_batch)
                cls_pred[i:] = temp.reshape((-1,))
            else:
                x_batch = x[i:j]
                _, temp = self.predict(x_batch)
                cls_pred[i:j] = temp.reshape((-1,))
            i = j
        cm = confusion_matrix(y_true=y_cls, y_pred=cls_pred)
        print(cm)

    def construct_graph(self):
        np.random.seed(self.np_seed)
        tf.set_random_seed(self.tf_seed)

        # network params from mlp_mnist_hwalsuklee
        # n_input = 784
        # n_hidden_1 = 256
        # n_hidden_2 = 256
        # n_classes = 10

        input_size = self.feature_dim
        output_size = self.y_train.shape[1]

        self.x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')
        self.y_cls = tf.placeholder(tf.int32, shape=[None, 1], name='y_cls')
        self.rate = tf.placeholder(tf.float32, shape=[], name='rate')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        w_shape = (784, 256)
        b_shape = (256, )
        w_name = "w1"
        b_name = "b1"
        self.w1, self.b1 = self.layer_w_and_b(w_shape, b_shape, w_name, b_name)
        self.x1 = self.fully_connected(self.x, self.w1, self.b1, batch_norm=False, activation=True, dropout=False)

        w_shape = (256, 256)
        b_shape = (256, )
        w_name = "w2"
        b_name = "b2"
        self.w2, self.b2 = self.layer_w_and_b(w_shape, b_shape, w_name, b_name)
        self.x2 = self.fully_connected(self.x1, self.w2, self.b2, batch_norm=False, activation=True, dropout=False)

        w_shape = (256, 10)
        b_shape = (10, )
        w_name = "w3"
        b_name = "b3"
        self.w3, self.b3 = self.layer_w_and_b(w_shape, b_shape, w_name, b_name)
        self.logits = tf.identity(self.fully_connected(self.x2, self.w3, self.b3, batch_norm=False, activation=False, dropout=False), name='logits')

        self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
        self.y_pred_cls = tf.cast(tf.argmax(self.y_pred, axis=1), tf.int32, name='y_pred_cls')

        self.entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.cost = tf.reduce_mean(self.entropy)
        
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

        self.correct_bool = tf.equal(self.y_pred_cls, self.y_cls, name='correct_bool')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32), name='accuracy')

    def fully_connected(self, x, w, b, batch_norm=False, activation=True, dropout=False):
        y = tf.matmul(x, w) + b
        if batch_norm:
            y = tf.contrib.layers.batch_norm(y, is_training=self.is_train, 
                                             center=True, scale=True, decay=0.9)
        if activation:
            y = self.activation(y)
        if dropout:
            y = tf.nn.dropout(y, rate=self.rate)
        return y

    def layer_w_and_b(self, w_shape, b_shape, w_name, b_name):
        w = tf.get_variable(w_name, shape=w_shape, initializer=self.initializer)
        b = tf.get_variable(b_name, initializer=tf.zeros(b_shape))
        return w, b

    def plot_weights(self):
        w = self.sess.run(self.W)
        w_min = np.min(w)
        w_max = np.max(w)

        plt.figure()
        fig, axes = plt.subplots(3, 4)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Only use the weights for the first 10 sub-plots.
            if i < 10:
                image = w[:, i].reshape(self.img_shape)
                ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
                ax.set_xlabel("Weights: {0}".format(i))

            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()
        plt.close()

    def plot_9_images_with_false_prediction(self, x, y, y_cls):
        images_false_prediction = []
        cls_true = []
        cls_pred = []
        num_false_prediction = 0
        i = 0
        while num_false_prediction < 9:
            feed_dict = {self.x: [x[i]], self.y: [y[i]], self.y_cls: [[y_cls[i]]], self.rate: 0.0, self.is_train: False}
            y_pred_cls, correct_bool = self.sess.run([self.y_pred_cls, self.correct_bool], feed_dict=feed_dict)
            if not correct_bool[0][0]:
                images_false_prediction.append(x[i])
                cls_true.append(self.cls_names[y_cls[i]])
                cls_pred.append(self.cls_names[y_pred_cls[0]])
                num_false_prediction += 1
            i += 1
        plot_many_images_2d(images=images_false_prediction, img_shape=self.img_shape, cls=cls_true, cls_pred=cls_pred)

    def predict(self, x):
        feed_dict = {self.x: x, self.rate: 0.0, self.is_train: False}
        return self.sess.run([self.y_pred, self.y_pred_cls], feed_dict=feed_dict)

    def restore(self):
        self.saver = tf.train.import_meta_graph(self.save_path + '.meta', clear_devices=True)
        self.saver.restore(sess=self.sess, save_path=self.save_path)

        self.x = self.sess.graph.get_tensor_by_name('x:0')
        self.y = self.sess.graph.get_tensor_by_name('y:0')
        self.y_cls = self.sess.graph.get_tensor_by_name('y_cls:0')
        self.rate = self.sess.graph.get_tensor_by_name('rate:0')
        self.is_train = self.sess.graph.get_tensor_by_name('is_train:0')
        self.y_pred = self.sess.graph.get_tensor_by_name('y_pred:0')
        self.y_pred_cls = self.sess.graph.get_tensor_by_name('y_pred_cls:0')
        self.correct_bool = self.sess.graph.get_tensor_by_name('correct_bool:0')
        self.accuracy = self.sess.graph.get_tensor_by_name('accuracy:0')

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

        grandient_step = 0
        for _ in range(self.epoch):
            x, y, y_cls = self.shuffle_data(self.x_train, self.y_train, self.y_train_cls)
            for i in range(self.x_train.shape[0] // self.batch_size):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
                y_batch_cls = y_cls[i * self.batch_size:(i + 1) * self.batch_size].reshape((-1, 1))
                feed_dict = {self.x: x_batch, self.y: y_batch, self.y_cls: y_batch_cls,
                             self.rate: self.drop_out_rate, self.is_train: True}
                grandient_step += 1
                if grandient_step % self.report_period == 0:
                    loss, _ = self.sess.run([self.cost, self.opt], feed_dict=feed_dict)
                    print('grandient_step : ', grandient_step)
                    print('loss :           ', loss)
                    print()
                    # self.plot_weights()
                else:
                    self.sess.run(self.opt, feed_dict=feed_dict)

        self.save()
