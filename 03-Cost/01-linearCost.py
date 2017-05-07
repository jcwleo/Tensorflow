# -*- coding: utf-8 -*-
import tensorflow as tf

x_data=[1.,2.,3.]                                       # x값 데이터 셋
y_data=[1.,2.,3.]                                       # y값 데이터 셋

W=tf.Variable(tf.random_uniform([1], -10.0,10.0))       # 트레이닝 하기위한 처음 랜덤 Weight 값

X=tf.placeholder(tf.float32)                            # X라는 변수에 실수형 값이 들어갈 공간을 만들어줌/ 트레이닝 할떄 값을 결정해줌
Y=tf.placeholder(tf.float32)                            # Y라는 변수에 실수형 값이 들어갈 공간을 만들어줌

hypothesis = W * X                                      # H(x)=WX, 여기서 X로 해준 이유는 나중에 직접 예측값을 알아보기 위해서

cost = tf.reduce_mean(tf.square(hypothesis - Y))        # 트레이닝 과정중 나온 값을 실제 y값과 비교를 하여 차이값의 제곱(양수여야하므로)의 평균을 구함


# 이 과정이 GradientDecent Algorithm 이다.
# Cost값의 기울기가 0일떄 최소값을 가지므로 가장 이상적인 W값을 구할수 있다
descent = W - tf.multiply(0.1, tf.reduce_mean(tf.multiply((tf.multiply(W,X)-Y),X)))    # Cost의 기울기 값을 0으로 만들기 위해,
                                                                        # 원래 기울기에서 과거의 기울기의 값에 rate값 a(0.1)을 곱한 값을 뺴줌

update =W.assign(descent)                               # W값을 새로 구한 값으로 업데이트


init = tf.global_variables_initializer()                    # 변수값 초기화

sess=tf.Session(config = tf.ConfigProto(device_count ={'GPU' : 0}))                                       # 실행
sess.run(init)
print ('step','cost','    W')
for step in range(100):

    print ('',step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))   # 반복하면서 step, cost, W값을 출력
    sess.run(update, feed_dict={X:x_data,Y:y_data})                                # 반복하면서 W값을 업데이트 시켜줌





