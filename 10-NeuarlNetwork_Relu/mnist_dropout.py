# -*- coding: utf-8 -*-

import tensorflow as tf
import input_data
import time

t1 = time.time() # start time

def xavier_init(n_inputs, n_ouptut, uniform=True):
    if uniform :
        init_range = tf.sqrt(6.0/(n_inputs + n_ouptut))
        return tf.random_uniform_initializer(-init_range, init_range)
    else :
        stddev = tf.sqrt(3.0/ (n_inputs + n_ouptut))
        return tf.truncated_normal_initializer(stddev=stddev)


mnist = input_data.read_data_sets("tmp", one_hot=True)

training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784])    # mnist 데이터가 28*28 사이즈로 있기 때문에 28*28=784 로 지정해준다
y = tf.placeholder("float", [None, 10])     # 0-9 사이의 숫자를 출력해주기 때문에 10 으로 지정해준다


# Set model weights
W1 = tf.get_variable("W1", shape=[784,500],initializer=xavier_init(784,500))
W2 = tf.get_variable("W2", shape=[500,256],initializer=xavier_init(500,256))
W3 = tf.get_variable("W3", shape=[256,128],initializer=xavier_init(256,128))
W4 = tf.get_variable("W4", shape=[128,10],initializer=xavier_init(128,10))


b1 = tf.Variable(tf.random_normal([500]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([128]))
b4 = tf.Variable(tf.random_normal([10]))


dropout_rate = tf.placeholder("float")

# Set model Layers
_L2=tf.nn.relu(tf.add(tf.matmul(x,W1),b1))  # 첫번째 레이어의 게이트들의 값을 두번째 레이어의 게이트들로 넘겨줌
L2=tf.nn.dropout(_L2,dropout_rate)          # 첫번째 레이어에서 dropout_rate 비율의 레이어만 남기고 dropout시킴
_L3=tf.nn.relu(tf.add(tf.matmul(L2,W2),b2)) # 두번째 레이어의 게이트들의 값을 세번째 레이어의 게이트들로 넘겨줌
L3=tf.nn.dropout(_L3,dropout_rate)          # 두번째 레이어에서 dropout_rate 비율의 레이어만 남기고 dropout시킴
_L4=tf.nn.relu(tf.add(tf.matmul(L3,W3),b3)) # 세번째 레이어의 게이트들의 값을 네번째 레이어의 게이트들로 넘겨줌
L4=tf.nn.dropout(_L4,dropout_rate)          # 세번째 레이어에서 dropout_rate 비율의 레이어만 남기고 dropout시킴

hypothesis = tf.add(tf.matmul(L4, W4),b4)   # 여기서 Softmax를 취하지 않음

# Minimize error using cross entropy
learning_rate = 0.001
# learning_rate = 10

# Cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis,y))    # Cost값을 구할때 Softmax를 취해주고 cost 값을 구함
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)    # GradientDecentOptimizer보다 성능이 좋은 AdamOptimizer를 사용

# Initializing the variable
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7})    # 학습을 시킬때 dropout_rate 값을 줌

            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7})/total_batch
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 실제로 테스트 할때는 dropout 시켰던 모든 게이트들을 이용해야하기 떄문에 dropout_rate = 1 로 한다
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate:1}))

t2 = time.time() # end time

print("execution time : ","{:5.3f}".format(t2-t1))
