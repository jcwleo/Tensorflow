# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np                                          # txt를 읽어와서 학습 시키기 위한 라이브러리

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')  # train.txt 파일을 배열로 읽어옴

x_data = xy[0:-1]                                           # train.txt 배열에서 0열부터 마지막열 전까지(b, x1, x2) 읽어옴
                                                            # (-1)은 마지막 열을 나타냄 [x:y]-> x열부터 y열 전까지
y_data = xy[-1]                                             # train.txt 배열에서 마지막열(y)을 읽어옴

print ('x', x_data)                                           # 읽어온 배열이 맞는지 확인
print ('y', y_data)

W=tf.Variable(tf.random_uniform([1,len(x_data)], -5.0,5.0)) # 트레이닝 하기위한 처음 랜덤 Weight 값 (1행,x데이터셋의 크기 만큼의 열)행렬로 나타냄
                                                            # 1열 값들이 bias의 랜덤값

hypothesis = tf.matmul(W, x_data)                           # H(x) = (W1 * x1) + (W2 * x2) + b / 'matmul' 함수를 이용하여 배열 곱을 해줌

cost = tf.reduce_mean(tf.square(hypothesis - y_data))       # 트레이닝 과정중 나온 값을 실제 y값과 비교를 하여 차이값의 제곱(양수여야하므로)의 평균을 구함

a = tf.Variable(0.1)                                        # cost값이 최소가 되게 하기 위한 작업
optimizer = tf.train.GradientDescentOptimizer(a)            # a = rate값 / GradientDescent Algorithm 함수를 사용
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()                        # 변수값 초기화

sess=tf.Session()                                           # 실행
sess.run(init)


print ('step','cost','    b','            W1','            W2')
for step in range(2001):
    sess.run(train)                                     # 2000번동안 트레이닝 시킴
    if step % 20 ==0:
        print (step, sess.run(cost), sess.run(W))

