# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

xy = np.loadtxt('xor_dataset.txt', unpack=True)

# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')


# 원래의 LogisticRegression으로는 XOR가 판별이 안되므로
# 두개의 입력 게이트를 통해서 판별을 한다.
W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='Weight1')      # 두개의 입력과 출력을 각각 내보낼 배열
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='Weight2')      # 첫번째 게이트에서 나온 두개의 출력값을 입력받아 한개의 출력으로 내보내는 배열

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

with tf.name_scope("Layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.name_scope("Layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

# Cost function
with tf.name_scope("Cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))
    cost_summ = tf.scalar_summary("Cost",cost)

# Minimize cost.
a = tf.Variable(0.1)
with tf.name_scope("Train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

w1_hist = tf.histogram_summary("weights1",W1)
w2_hist = tf.histogram_summary("weights2",W2)

b1_hist = tf.histogram_summary("biases1",b1)
b2_hist = tf.histogram_summary("biases2",b2)

y_hist = tf.histogram_summary("y",Y)




# Initializa all variables.
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # tensorboard merge
    sess.run(init)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/xor_logs", sess.graph)

    # Run graph.
    for step in range(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            summary, _ = sess.run([merged, train], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))