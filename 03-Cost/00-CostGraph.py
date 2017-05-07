# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt                             # 그래프를 그리기 위한 라이브러리

X = [1., 2., 3.]
Y= [1., 2., 3.,]
m=n_samples = len(X)                                        # 평균값을 구하기 위한 데이터 셋의 갯수 값

W=tf.placeholder(tf.float32)                                # 트레이닝 하기위한 처음 랜덤 Weight 값

hypothesis = tf.multiply(X, W)                                   # H(x)=Wx
error = hypothesis - Y
cost = tf.reduce_mean(tf.sqrt(1+tf.square(error))-1)
         # 나온 값의 제곱의 합의 평균(Cost)을 구하기 위한 함수 사용

init = tf.global_variables_initializer()                        # 변수값 초기화

W_val =[]                                                   # 그래프를 그리기 위한 변수
cost_val=[]

sess=tf.Session(config = tf.ConfigProto(device_count ={'GPU' : 0}))                                           # 실행
sess.run(init)

for i in range(-30, 50):
    print (i*0.1, sess.run(cost, feed_dict={W: i*0.1}) )      # x축의 W값과 평균값(Cost)값을 출력
    W_val.append(i*0.1)                                     # x축의 값을 입력
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))   # y축의 값을 입력

plt.plot(W_val, cost_val, 'ro')                             # 그래프 출력
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()