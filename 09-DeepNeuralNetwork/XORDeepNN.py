# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

xy = np.loadtxt('xor_dataset.txt', unpack=True)

# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([3, 4], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([3]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")


# 일반 NN 보다 더 많은 레이어를 갖는다.

#  레이어1         레이어2        레이어3
# x1 -      |   s1 -        |
#      s1   |   s2 -  ss1   |
# x2 -      |   s3 -        |
#           |               |
# x1 -      |   s1 -        |     ss1 -
#      s2   |   s2 -  ss2   |     ss2 -    sss1 = Hypothesis
# x2 -      |   s3 -        |     ss3 -
#           |               |     ss4 -
# x1 -      |   s1 -        |
#      s3   |   s2 -  ss3   |
# x2 -      |   s3 -        |
#           |               |
#           |   s1 -        |
#           |   s2 -  ss4   |
#           |   s3 -        |


# Hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

# Cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))

# Minimize cost.
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initializa all variables.
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session(config = tf.ConfigProto(device_count ={'GPU' : 0})) as sess:
    sess.run(init)

    for step in range(8001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(
                step,
                sess.run([cost,cost], feed_dict={X: x_data, Y: y_data}),
                sess.run(W1),
                sess.run(W2)
            )

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))