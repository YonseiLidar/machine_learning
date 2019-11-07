import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


class LR:
    
    def __init__(self, sess, data, 
                 lr=5e-1, epoch=int(2e2), batch_size=100,
                 initializer=tf.keras.initializers.constant(1.0)):
        
        self.sess = sess
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_train_cls, self.y_test_cls = data
        
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.initializer = initializer
        
        self.x = tf.placeholder(tf.float32, shape=(None, 64))
        self.y = tf.placeholder(tf.float32, shape=(None, 10))  # 0001000000
        self.y_cls = tf.placeholder(tf.int32, shape=(None,))  # 3

        self.beta = tf.get_variable('beta', (64, 10), dtype=tf.float32, initializer=self.initializer)

        self.logits = self.x @ self.beta  # (?,10) = (?,64) @ (64,10)
        self.entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)  # (?,)
        self.loss = tf.reduce_mean(self.entropy)  # ()
        self.op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        self.y_pred_prob = tf.nn.softmax(self.logits)
        self.y_pred_cls = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
        self.correct_bool = tf.equal(self.y_pred_cls, self.y_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32))

        self.beta0_trace = []
        self.beta1_trace = []
        self.beta2_trace = []
        self.beta3_trace = []
        self.loss_trace = []

    def compute_test_accuracy(self):

        feed_dict = {self.x: self.x_test, self.y: self.y_test, self.y_cls: self.y_test_cls}
        test_accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
        print("test_accuracy : ", test_accuracy)

        y_test_pred_cls = self.sess.run(self.y_pred_cls, feed_dict=feed_dict)
        cm = confusion_matrix(self.y_test_cls, y_test_pred_cls)
        print(cm)
        
    def train(self):
        
        self.sess.run(tf.global_variables_initializer())
    
        gradient_step = -1
        for i in range(self.epoch):
            idx = np.arange(self.x_train.shape[0])
            np.random.shuffle(idx)
            x_data = self.x_train[idx]
            y_data = self.y_train[idx]
            for j in range((self.x_train.shape[0]//self.batch_size)-1):
                gradient_step += 1
                feed_dict = {self.x: x_data[j*self.batch_size:(j+1)*self.batch_size],
                             self.y: y_data[j*self.batch_size:(j+1)*self.batch_size]}
            if gradient_step == 0:
                beta_run, loss_run = self.sess.run([self.beta, self.loss], feed_dict=feed_dict)
                self.beta0_trace.append(beta_run[23, 7])
                self.beta1_trace.append(beta_run[15, 7])
                self.beta2_trace.append(beta_run[55, 7])
                self.beta3_trace.append(beta_run[33, 7])
                self.loss_trace.append(loss_run)
            elif gradient_step % 100 == 0:
                beta_run, loss_run, _ = self.sess.run([self.beta, self.loss, self.op], feed_dict=feed_dict)
                self.beta0_trace.append(beta_run[23, 7])
                self.beta1_trace.append(beta_run[15, 7])
                self.beta2_trace.append(beta_run[55, 7])
                self.beta3_trace.append(beta_run[33, 7])
                self.loss_trace.append(loss_run)
            else:
                self.sess.run(self.op, feed_dict=feed_dict)