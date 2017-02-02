# x = 1
# y = x + 9
# print y
import tensorflow as tf
x = tf.constant(1, name='x')
y = tf.Variable(x+9, name='y')
# print y
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print session.run(y)


a = tf.placeholder("int32")
b = tf.placeholder("int32")
y = tf.mul(a, b)
sess = tf.Session()
print sess.run(y, feed_dict={a: 2, b: 5})
