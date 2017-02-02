import tensorflow as tf
import matplotlib.pyplot as plt

"""uniform distribution"""
uniform = tf.random_uniform([100], minval=0, maxval=1, dtype=tf.float32)
sess = tf.Session()
with tf.Session() as session:
    print uniform.eval()
    plt.hist(uniform.eval(), normed=True)
    plt.show()

"""normal distribution"""
norm = tf.random_normal([100], mean=0, stddev=2)
with tf.Session() as session:
    plt.hist(norm.eval(), normed=True)
    plt.show()