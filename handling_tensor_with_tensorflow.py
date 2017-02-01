import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow as tf
filename = "images/1.jpg"
input_image = mp_image.imread(filename)
print "input dim = {}".format(input_image.ndim)
print "input shape = {}".format(input_image.shape)

"""slice is a bidimensional segment of the starting image"""
# my_image = tf.placeholder("uint8", [None, None, 3])
# slice = tf.slice(my_image, [10, 0, 0], [16, -1, -1])
# with tf.Session() as sess:
#     result = sess.run(slice, feed_dict={my_image: input_image})
#     print result.shape
# plt.imshow(result)
# plt.show()

"""This method performs a swap between the axes 0 and 1 of the input matrix, while the z axis is left unchanged"""
x = tf.Variable(input_image, name='x')
model = tf.initialize_all_variables()
with tf.Session() as sess:
    x = tf.transpose(x, perm=[1, 0, 2])
    sess.run(model)
    result = sess.run(x)
plt.imshow(result)
plt.show()
