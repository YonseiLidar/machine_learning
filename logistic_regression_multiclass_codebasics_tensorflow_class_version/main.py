from data_loading import data_loader
from model import logistic_regression
from utils import plot_trace
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

data = data_loader.data_digits()

with tf.Session() as sess:
    a = logistic_regression.LR(sess, data,
                               lr=1e-2,
                               epoch=int(1e4),
                               batch_size=100)
    a.train()
    a.compute_test_accuracy()
    trace_data = (a.beta0_trace, a.beta1_trace, a.beta2_trace, a.beta3_trace, a.loss_trace)

plot_trace.trace_plot(trace_data)