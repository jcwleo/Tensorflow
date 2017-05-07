# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

xy = np.loadtxt('xor_dataset.txt', unpack=True)

# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# 원래의 LogisticRegression으로는 XOR가 판별이 안되므로
# 두개의 입력 게이트를 통해서 판별을 한다.
W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))      # 두개의 입력과 출력을 각각 내보낼 배열
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))      # 첫번째 게이트에서 나온 두개의 출력값을 입력받아 한개의 출력으로 내보내는 배열

b1 = tf.Variable(tf.zeros([2]), name="Bias1")               # 첫번째 두개의 게이트의 각각의 BIAS 값
b2 = tf.Variable(tf.zeros([1]), name="Bias2")               # 두번째 게이트의 바이어스 값

# x1 -
#      (x1 * W1[0,0]) + (x2 * W1[1,0]) + b1 -> Sigmoid = S1
# x2 -
#                                                          (S1 * W2[0,0]) + (S2 * W2[1,0]) + b2 -> Sigmoid = Hypothesis
# x1 -
#      (x1 * W1[0,1]) + (x2 * W1[1,1]) + b1 -> Sigmoid = S2
# x2 -

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
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)    # tf.floor(x) : x의 값보다 크지 않은 정수를 출력한다
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print('-' * 20)
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],  # 트레이닝된 H값을 출력
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))