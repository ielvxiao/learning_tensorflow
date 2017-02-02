import tensorflow as tf
"""seed=1,THis means that repeatedly evaluating the two distributions"""
uniform_with_seed = tf.random_uniform([1], seed=1)
uniform_without_seed = tf.random_uniform([1])
print "First Run"
with tf.Session() as first_session:
    print "uniform with (seed=1) = {}"\
        .format(first_session.run(uniform_with_seed))
    print "uniform with (seed=1) = {}" \
        .format(first_session.run(uniform_with_seed))
    print "unifrom without seed={}"\
    .format(first_session.run(uniform_without_seed))
