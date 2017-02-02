import tensorflow as tf
import matplotlib.pyplot as plt
trials = 1000
hists = 0
x = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)
y = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)
pi = []
sess = tf.Session()
with sess.as_default():
    for i in range(1, trials):
        for j in range(1, trials):
            if x.eval()**2 + y.eval()**2 <1 :
                hists += 1
                pi.append((4 * float(hists) / i) / trials)

plt.plot(pi)
plt.show()