# -*- coding: utf-8 -*-
import tensorflow as tf

x_data=[[1., 0., 3., 0., 5.],                           # 행렬(벡터)을 이용해 x데이터 셋을 나타냄
        [0., 2., 0., 4., 0.]]

y_data=[1., 2., 3., 4., 5.]                             # y값 데이터 셋

W=tf.Variable(tf.random_uniform([1,2], -1.0,1.0))       # 트레이닝 하기위한 처음 랜덤 Weight 값 (1행,2열)행렬로 나타냄

b=tf.Variable(tf.random_uniform([1], -1.0,1.0))         # 트레이닝 하기위한 처음 랜덤 Bias 값

hypothesis = tf.matmul(W, x_data) + b                   # H(x) = (W1 * x1) + (W2 * x2) + b / matmul 함수를 이용하여 배열 곱을 해줌

cost = tf.reduce_mean(tf.square(hypothesis - y_data))   # 트레이닝 과정중 나온 값을 실제 y값과 비교를 하여 차이값의 제곱(양수여야하므로)의 평균을 구함

a = tf.Variable(0.1)                                    # cost값이 최소가 되게 하기 위한 작업
optimizer = tf.train.GradientDescentOptimizer(a)        # a = rate값
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()                    # 변수값 초기화

sess=tf.Session()                                       # 실행
sess.run(init)


print ('step','cost','    W1','            W2','            b')
for step in range(2001):
    sess.run(train)                                     # 2000번동안 트레이닝 시킴
    if step % 20 ==0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))

