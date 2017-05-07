# -*- coding: utf-8 -*-
from random import randint

import matplotlib.pyplot as plt
import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("tmp", one_hot=True)

training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist 데이터가 28*28 사이즈로 있기 때문에 28*28=784 로 지정해준다
y = tf.placeholder("float", [None, 10]) # 0-9 사이의 숫자를 출력해주기 때문에 10 으로 지정해준다


# Set model weights
W = tf.Variable(tf.zeros([784, 10])) # mnist 데이터 하나를 0~9 사이의 숫자로 지정할 수 있게 배열을 만든다
b = tf.Variable(tf.zeros([10]))

# Construct Model
# https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginnners/index.html
# First, we multiply x by W with the expression tf.matmul(x, W).
# This is flipped from when we multiplied them in our equation,
# where we had Wx, as a small trick to deal with x being
# a 2D tensor with multiple inputs.
# We then add b, and finally apply tf.nn.softmax
#hypothesis = tf.nn.softmax(tf.matmul(X, W)) # Softmax
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
learning_rate = 0.001
# learning_rate = 10

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variable
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        # 많은 데이터 중에 쪼개서 25개의 데이터 그룹만 사용(training_epochs = 25)
        # (training set of 60,000 examples, and a test set of 10,000 examples)
        # 600장 * 25번을 트레이닝 함
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)      # 받아온 6만장의 샘플을 한번에 학습할 양으로 나눠줌
                                                                    # 60000/100 = 600

        for i in range(total_batch):                                # 600장을 트레이닝함줌
            # batch_xs와 batch_ys에 각각 728사이즈의 이미지와 그에 맞는 라벨값을 넣어줌
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))


    print ("Optimization Finished!")

    # Get one and predict
    r = randint(0, mnist.test.num_examples - 1)
    print ("Label: ", sess.run(tf.arg_max(mnist.test.labels[r:r+1], 1)))  #테스트 하기위해 랜덤한 것을 가져와서 예측해봄
    print ("Prediction: ", sess.run(tf.arg_max(activation, 1), {x: mnist.test.images[r:r+1]}))

    # Show the Image
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()