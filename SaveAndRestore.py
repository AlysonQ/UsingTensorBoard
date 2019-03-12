#!/usr/bin/python
# -*- coding: utf-8 -*-
# example of how to save and restore tensorflow
# Alyson Chen(qoo810823@gmail.com)
# 2019.03.10 in Taiwan
import tensorflow as tf
import numpy as np
from datetime import datetime

#定義TensorBoard名稱與時間戳記
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

## Save to file
# remember to define the same dtype and shape when restore
weights = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
biases = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)

# 這裡是讀取的部分------START
# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
#
# #若只是要讀取就不用做初始化變量的動作
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     # 提取变量
#     saver.restore(sess, "my_net/save_net.ckpt")
#     print("weights:", sess.run(W))
#     print("biases:", sess.run(b))
# 這裡是讀取的部分------END


#將運行的結果寫入到TensorBoard裡面
tf.summary.merge_all() #將所有要顯示在TensorBorad的資料整合
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())#將資料寫進log黨裡，並且儲存至指定的路徑，這裡是logdir定義的路徑

#記得要關閉才能存擋
file_writer.close()
先建立 W, b 的容器
