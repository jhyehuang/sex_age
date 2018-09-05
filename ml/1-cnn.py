
# coding: utf-8

# In[1]:


"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


# 我们在这里调用系统提供的Mnist数据函数为我们读入数据，如果没有下载的话则进行下载。
# 
# <font color=#ff0000>**这里将data_dir改为适合你的运行环境的目录**</font>

# In[2]:


# Import data
data_dir = '/tmp/tensorflow/mnist/input_data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)


# 一个非常非常简陋的模型

# In[3]:


# Create the model
x = tf.placeholder(tf.float32, [None, 784])
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 22])
# variables


# ## 权重初始化
# 权重初始化时，将标准差从0.1调整为0.71

# In[4]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# # 卷积和池化

# In[5]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# # 第一层卷积

# In[7]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# # 第二层卷积

# In[8]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# # 密集连接层
# 999  98.66
# 784  98.11
# 

# In[42]:


W_fc1 = weight_variable([7 * 7 * 64, 999])
b_fc1 = bias_variable([999])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# #  Dropout

# In[43]:


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# # 输出层

# In[44]:


W_fc2 = weight_variable([999, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# # 训练和评估模型

# ## 计算交叉熵

# In[49]:


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4*0.9).minimize(cross_entropy) 
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[50]:


with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(25000):
        batch = mnist.train.next_batch(50)
        if i%100==0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print ("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# ### 使用tensorflow，构造并训练一个神经网络，在测试机上达到超过98%的准确率。
# 在完成过程中，需要综合运用目前学到的基础知识：
# - 深度神经网络
# - 激活函数
# - 正则化
# - 初始化
# - 卷积
# - 池化
# 
# 
# ### 并探索如下超参数设置：
# - 卷积kernel size
# - 卷积kernel 数量
# - 学习率
# - 正则化因子
# - 权重初始化分布参数整

