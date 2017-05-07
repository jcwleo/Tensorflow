# -*- coding: utf-8 -*-
import tensorflow as tf

x_data=[1.,2.,3.]                                       # x값 데이터 셋
y_data=[1.,2.,3.]                                       # y값 데이터 셋

W=tf.Variable(tf.random_uniform([1], -3.0,3.0))         # 트레이닝 하기위한 처음 랜덤 Weight 값
b=tf.Variable(tf.random_uniform([1], -3.0,3.0))         # 트레이닝 하기위한 처음 랜덤 Bias 값

X=tf.placeholder(tf.float32)                            # X라는 변수에 실수형 값이 들어갈 공간을 만들어줌/ 트레이닝한 W로 H값을 구하기 위해서
Y=tf.placeholder(tf.float32)                            # Y라는 변수에 실수형 값이 들어갈 공간을 만들어줌

hypothesis = W * X + b                                  # H(x)=Wx+b, 여기서 X로 해준 이유는 나중에 직접 예측값을 알아보기 위해서

cost = tf.reduce_mean(tf.square(hypothesis - Y))        # 트레이닝 과정중 나온 값을 실제 y값과 비교를 하여 차이값의 제곱(양수여야하므로)의 평균을 구함

a= tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)                          # cost값이 최소가 되게 하기 위한 작업

init = tf.global_variables_initializer()                    # 변수값 초기화

sess=tf.Session()                                       # 실행
sess.run(init)
print ('step','cost','    W','            b')
for step in range(2001):
    sess.run(train, feed_dict={X:x_data,Y:y_data})      # 2000번동안 원래 데이터셋을 이용해 Weight값을 구함
    if step % 20 ==0:
        print ('',step, sess.run(cost, feed_dict={X:x_data,Y:y_data}), sess.run(W), sess.run(b))

print (sess.run(hypothesis,feed_dict={X:5.})  )           # 트레이닝 시킨 Weight값으로 h값을 구함
print (sess.run(hypothesis,feed_dict={X:2.5}))

