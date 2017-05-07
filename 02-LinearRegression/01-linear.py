# -*- coding: utf-8 -*-
import tensorflow as tf

x_data=[1.,2.,3.]                                       # x값 데이터 셋
y_data=[1.,2.,3.]                                       # y값 데이터 셋
model_path = "save/model.ckpt"
W=tf.Variable(tf.random_uniform([1], -1.0,1.0))         #  트레이닝 하기위한 처음 랜덤 Weight 값
b=tf.Variable(tf.random_uniform([1], -1.0,1.0))         #트레이닝 하기위한 처음 랜덤 Bias 값

hypothesis = W * x_data + b                             # H(x)=Wx+b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))   # 트레이닝 과정중 나온 값을 실제 y값과 비교를 하여 차이값의 제곱(양수여야하므로)의 평균을 구함

a= tf.Variable(0.00025)
optimizer = tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)                          # cost값이 최소가 되게 하기 위한 작업

init = tf.global_variables_initializer()                    # 변수값 초기화

sess=tf.Session()                                       # 실행
saver = tf.train.Saver()
sess.run(init)


print ('step','cost','    W','            b')
for step in range(2001):
    sess.run(train)                                     # 2000번동안 트레이닝 시킴
    if step % 20 ==0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))

save_path = saver.save(sess, model_path)
print("Model saved in file: ",save_path)
