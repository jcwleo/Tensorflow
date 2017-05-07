# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

xy = np.loadtxt('xor_dataset.txt', unpack=True)

# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([10]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")


# 일반 NN과 다른점은 더 넓게 많은 수의 뉴럴을 가질 수 있다.
# 더 정확한 트레이닝이 가능하다.

# x1 -
#      (x1 * W1[0,0]) + (x2 * W1[1,0]) + b1 -> Sigmoid = S1
# x2 -
#
# x1 -
#      (x1 * W1[0,1]) + (x2 * W1[1,1]) + b1 -> Sigmoid = S2
# x2 -
#
# x1 -
#      (x1 * W1[0,2]) + (x2 * W1[1,2]) + b1 -> Sigmoid = S3
# x2 -
#
# x1 -
#      (x1 * W1[0,3]) + (x2 * W1[1,3]) + b1 -> Sigmoid = S4
# x2 -
#                                                  (S1 * W2[0,0]) + (S2 * W2[1,0]) + ... +  b2 -> Sigmoid = Hypothesis
# x1 -
#      (x1 * W1[0,4]) + (x2 * W1[1,4]) + b1 -> Sigmoid = S5
# x2 -
#
# x1 -
#      (x1 * W1[0,5]) + (x2 * W1[1,5]) + b1 -> Sigmoid = S6
# x2 -
#               .
#               .
#               .



# Hypotheses
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

# Cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))

# Minimize cost.
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initializa all variables.
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for step in range(8001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(
                step,
                sess.run(cost, feed_dict={X: x_data, Y: y_data}),
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