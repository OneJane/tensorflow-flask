import os

import input_data
import model
import tensorflow as tf

# 从input_data中下载数据到MNIST_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# create model
with tf.variable_scope("regression"):
    # 用户输入占位符
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 训练步骤
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 预测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 保存训练变量参数
saver = tf.train.Saver(variables)
# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(20000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 打印测试集和训练集的精准度
    print((sess.run(accuracy, feed_dict={x:data.test.images, y_:data.test.labels})))

    # 保存训练好的模型
    path = saver.save(
        sess,os.path.join(os.path.dirname(__file__),'data','regression.ckpt'),
        write_meta_graph=False,write_state=False)
    print("Saved:", path)