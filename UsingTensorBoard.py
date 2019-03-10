#!/usr/bin/python
# -*- coding: utf-8 -*-
# How to use TensorBoard example
# Alyson Chen(qoo810823@gmail.com)
# 2019.03.10 in Taiwan
import tensorflow as tf
from datetime import datetime

#定義TensorBoard名稱與時間戳記
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#面積的公式等於長乘以寬
width = tf.placeholder("int32",name='width')
height = tf.placeholder("int32",name='height')
area = tf.multiply(width,height,name='area')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('[Alyson log] area=',sess.run(area,feed_dict={width: 6,height: 8}))

#將運行的結果寫入到TensorBoard裡面
tf.summary.merge_all() #將所有要顯示在TensorBorad的資料整合
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())#將資料寫進log黨裡，並且儲存至指定的路徑，這裡是logdir定義的路徑

#記得要關閉才能存擋
file_writer.close()
