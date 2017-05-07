# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np  # txt를 읽어와서 학습 시키기 위한 라이브러리

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')  # train.txt 파일을 배열로 읽어옴

x_data = np.transpose(xy[0:3])  # train.txt 배열에서 0열부터 2열까지(b, x1, x2) 읽어옴
# (-1)은 마지막 열을 나타냄 [x:y]-> x열부터 y열 전까지
y_data = np.transpose(xy[3:])  # train.txt 배열에서 3열부터 5열까지 (y(등급))을 읽어옴 / one - hot encoding

print ('x', x_data)  # 읽어온 배열이 맞는지 확인
print ('y', y_data)

X = tf.placeholder("float", [None, 3])  # 행의 크기는 무제한, 열의 크기는 3개(b, x1, x2)짜리 빈 공간을 만들어줌
Y = tf.placeholder("float", [None, 3])  # 행의 크기는 무제한, 열의 크기는 3개(A, B, C)짜리 빈 공간을 만들어줌

W = tf.Variable(tf.zeros([3, 3]))  # Weight 값들이 들어갈 공간[3, 3]을 0으로 채움
# LogisticRegression에서는 두가지(0 or 1) 만 구별하면 되기 때문에
# [1, 3] 행렬이지만 SoftmaxRegression에서는 여러개의 조건이 있으므로
# 조건만큼의 행을 만들어 준다. 여기서는 A, B, C 세개의 등급을 구별하기 때문에
# [3, 3]의 행렬을 만들어준다.


hypothesis = tf.nn.softmax(tf.matmul(X, W))  # 두개의 행렬을 곱해준것에다가 Softmax 를 해준다
# 나온 값을 S(x)= e^x / sigma(j, e^j) 를 해줘야
# 각각의 값의 확률분포를 알수 있다
# Tensorflow 에서는 softmax 라는 함수를 이용해 구할 수 있다

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

# Cross-entropy cost function 을 이용해 코스트 값들의 합의 평균을 구한다
# -log(hypothesis) 그래프는 0에서는 무한대에 가깝고, 1에서는 0인 그래프이다.

learning_rate = 0.2  # GradientDecent Algolithm에서 얼만한 step으로 내려갈 것인지
# learning_rate 로 값을 정해준다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()  # 변수값 초기화

with tf.Session() as sess:  # 실행
    sess.run(init)

    for step in range(10001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})  # 2000번동안 트레이닝 시킴
        if step % 200 == 0:
            print (step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


    print ('')
    a= sess.run(hypothesis, feed_dict={X:[[1,11,7]]})
    print (a, sess.run(tf.arg_max(a,1)))

    b= sess.run(hypothesis, feed_dict={X:[[1,5,5]]})
    print (b, sess.run(tf.arg_max(b,1)))

    c= sess.run(hypothesis, feed_dict={X:[[1,1,0]]})
    print (c, sess.run(tf.arg_max(c,1)))

    all= sess.run(hypothesis, feed_dict={X:[[1,11,7],[1,3,4],[1,1,0]]})
    print (all, sess.run(tf.arg_max(all,1)))