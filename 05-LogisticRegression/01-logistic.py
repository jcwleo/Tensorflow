# -*- coding: utf-8 -*-


# 0과 1 둘중 하나를 판별해주는 알고리즘 ex)이메일이 스펨인지 아닌지, 시험을 패스했는지 못했는지 알수있음
# 일반적인 LinearRegression으로 학습을 하면 X값이 커지면 기울기 값이 크게 바뀌고 기준의 값이 바뀐다.
# 값이 바뀌게 되면 원래 판별됬던 값을의 판별값들이 바뀌게 된다(기준이 바뀌기 때문에)
# 그러므로 LogisticClassification 으로 구현해야 한다.
# H(x)=1 / (1 + e^(-WX)) 를 이용하면 최대값은 1에 수렴 최소값은 0에 수렴하게 된다.
# Cost 에서는 원래의 Cost Function을 이용하게 되면 그래프 모양이 구불구불 해지기 때문에 사용할수 없다.
# 그래서 Log 함수를 취해주어서 Cost 값(예측한 값과 실제값의 차이)을 구한다.

import tensorflow as tf
import numpy as np                                          # txt를 읽어와서 학습 시키기 위한 라이브러리

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')  # train.txt 파일을 배열로 읽어옴

x_data = xy[0:-1]                                           # train.txt 배열에서 0열부터 마지막열 전까지(b, x1, x2) 읽어옴
                                                            # (-1)은 마지막 열을 나타냄 [x:y]-> x열부터 y열 전까지
y_data = xy[-1]                                             # train.txt 배열에서 마지막열(y)을 읽어옴

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

print ('x', x_data)                                           # 읽어온 배열이 맞는지 확인
print ('y', y_data)

W=tf.Variable(tf.random_uniform([1,len(x_data)], -1.0,1.0)) # 트레이닝 하기위한 처음 랜덤 Weight 값 (1행,x데이터셋의 크기 만큼의 열)행렬로 나타냄
                                                            # 1열 값들이 bias의 랜덤값

h = tf.matmul(W, X)                                         # h = (W1 * x1) + (W2 * x2) + b / 'matmul' 함수를 이용하여 배열 곱을 해줌

hypothesis = tf.div(1.,1. + tf.exp(-h))                     # H(x)=1 / (1 + e^(-h)) H(x)의 크기 범위가 0~1 로 축소됨


cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis)) # Y값이 1이면 -log(H(x)), 0이면 -log(1-H(x))

a = tf.Variable(0.1)                                        # cost값이 최소가 되게 하기 위한 작업
optimizer = tf.train.GradientDescentOptimizer(a)            # a = rate값 / GradientDescent Algorithm 함수를 사용
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()                        # 변수값 초기화

sess=tf.Session()                                           # 실행
sess.run(init)


print ('step','cost','    b','            W1','            W2')
for step in range(2001):
    sess.run(train, feed_dict= {X:x_data, Y:y_data})                                     # 2000번동안 트레이닝 시킴
    if step % 20 ==0:
        print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))


print ('----------------------------------------------------------------')
# 학습 시킨값으로 공부한 시간과 참석횟수를 넣어 예측값을 도출
print (sess.run(hypothesis, feed_dict={X:[[1],[2],[2]]}) > 0.5, sess.run(hypothesis, feed_dict={X:[[1],[2],[2]]})) # 판별한 값이 절반(0.5)을 넘어가면 합격
print (sess.run(hypothesis, feed_dict={X:[[1],[5],[5]]}) > 0.5, sess.run(hypothesis, feed_dict={X:[[1],[5],[5]]}))
print (sess.run(hypothesis, feed_dict={X:[[1, 1],[4, 3],[3, 5]]}) > 0.5, sess.run(hypothesis, feed_dict={X:[[1, 1],[4, 3],[3, 5]]}))  # 여러면 동시에 확인 가능
