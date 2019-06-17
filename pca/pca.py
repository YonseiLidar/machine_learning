import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import utils.utils as utils


class PCA:
    
    def __init__(self, x_train, sess, dim_z=2, 
                 lr=1e-3, epoch=100, batch_size=100, report_period=1000,
                 np_seed=1, tf_seed=1, save_path='pca/model/pca_1'):
        self.x_train = x_train
        self.sess = sess
        self.dim_z = dim_z
        self.lr = lr
        self.epoch = epoch 
        self.batch_size = batch_size
        self.report_period = report_period
        self.np_seed = np_seed
        self.tf_seed = tf_seed
        self.save_path = save_path
        
        self.feature_dim = self.x_train.shape[1] 
        self.save_dir = self.save_path.split('/')[0] + '/' + self.save_path.split('/')[1]

        self.x = None
        self.E_W = None
        self.E_b = None
        self.z = None
        self.D_W = None
        self.D_b = None
        self.x_recon = None
        self.cost = None
        self.opt = None
        self.saver = None
    
    def construct_graph(self):
        
        np.random.seed(self.np_seed)
        tf.set_random_seed(self.tf_seed)
        
        input_size = self.feature_dim 
        
        self.x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        
        self.E_W = tf.get_variable("E_W",
                                   shape=[input_size, self.dim_z],
                                   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
        self.E_b = tf.get_variable("E_b", initializer=tf.zeros([self.dim_z]))
        self.z = tf.identity(tf.matmul(self.x, self.E_W) + self.E_b, name='z')
        
        self.D_W = tf.get_variable("D_W",
                                   shape=[self.dim_z, input_size],
                                   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
        self.D_b = tf.get_variable("D_b", initializer=tf.zeros([input_size]))
        self.x_recon = tf.identity(
            tf.matmul(self.z, self.D_W) + self.D_b, 
            name='x_recon')
        
        self.cost = tf.nn.l2_loss(self.x - self.x_recon)
        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.cost)
        
    def plot_16_generated(self, figure_save_dir='pca/img', figure_index=0):
        if not os.path.exists(figure_save_dir):
            os.makedirs(figure_save_dir)

        z = np.random.normal(0., 1., size=(16, self.dim_z))
        feed_dict = {self.z: z}
        images = self.sess.run(self.x_recon, feed_dict=feed_dict)
        
        n = images.shape[1]
        m = int(np.sqrt(n))
        img_shape = (m, m)

        fig = utils.plot_16_images_2d_and_return(
            images, img_shape=img_shape)
        plt.savefig(
            figure_save_dir + '/{}.png'.format(figure_index), 
            bbox_inches='tight')
        plt.close(fig)
        
    def plot_16_loading_vectors(self):
        z = np.zeros(shape=(16, self.dim_z))
        for i in range(16):
            if i < self.dim_z:
                z[i, i] = 1

        feed_dict = {self.z: z}
        images = self.sess.run(self.x_recon, feed_dict=feed_dict)
        
        n = images.shape[1]
        m = int(np.sqrt(n))
        img_shape = (m, m)

        fig = utils.plot_16_images_2d_and_return(
            images, img_shape=img_shape)
        plt.show(fig)
        
    def plot_16_original_and_recon(self, imgs_original):
        n = imgs_original.shape[1]
        m = int(np.sqrt(n))
        img_shape = (m, m)
        
        fig = utils.plot_16_images_2d_and_return(
            imgs_original, img_shape=img_shape)
        plt.show(fig)
        
        imgs_recon = self.recon(imgs_original)
        fig = utils.plot_16_images_2d_and_return(
            imgs_recon, img_shape=img_shape)
        plt.show(fig)
            
    def recon(self, x):
        feed_dict = {self.x: x}
        return self.sess.run(self.x_recon, feed_dict=feed_dict)
    
    def restore(self):
        self.saver = tf.train.import_meta_graph(self.save_path + '.meta',
                                                clear_devices=True)
        self.saver.restore(sess=self.sess, save_path=self.save_path)

        self.x = self.sess.graph.get_tensor_by_name('x:0')
        self.x_recon = self.sess.graph.get_tensor_by_name('x_recon:0')
        
    def save(self):
        self.saver = tf.train.Saver()

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver.save(sess=self.sess, save_path=self.save_path) 
        
    def shuffle_data(self, x):
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx]
    
    def train(self):
        self.construct_graph()
        tf.global_variables_initializer().run()

        gradient_step = 0
        for epoch_num in range(self.epoch):
            x = self.shuffle_data(self.x_train)
            for i in range(self.x_train.shape[0] // self.batch_size):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size] 
                feed_dict = {self.x: x_batch}
                
                gradient_step += 1
                if gradient_step % self.report_period == 0:
                    loss, _ = self.sess.run(
                        [self.cost, self.opt], 
                        feed_dict=feed_dict)
                    print('gradient_step : ', gradient_step)
                    print('loss :           ', loss)
                    print()
                    self.plot_16_generated(figure_index=gradient_step)
                else: 
                    self.sess.run(self.opt, feed_dict=feed_dict)
        
        self.save()


class PCANoisyBowl(PCA):

    def train(self):
        self.construct_graph()
        tf.global_variables_initializer().run()

        gradient_step = 0
        for epoch_num in range(self.epoch):
            x = self.shuffle_data(self.x_train)
            for i in range(self.x_train.shape[0] // self.batch_size):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.x: x_batch}

                gradient_step += 1
                if gradient_step % self.report_period == 0:
                    loss, _ = self.sess.run(
                        [self.cost, self.opt],
                        feed_dict=feed_dict)
                    print('gradient_step : ', gradient_step)
                    print('loss :           ', loss)
                    print()
                else:
                    self.sess.run(self.opt, feed_dict=feed_dict)

        self.save()