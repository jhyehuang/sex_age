
# coding: utf-8

# In[2]:


"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
sys.path.append('..')
from data_preprocessing import *
from flags import FLAGS, unparsed


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


if FLAGS.del_maxmin_mod:
    remove(FLAGS.tmp_data_path+'MinMaxScaler_model.joblib_dat')
if FLAGS.del_pca_mod:
    remove(FLAGS.tmp_data_path+'PCA_model.joblib_dat')
    

# Import data
x_train,y_train,x_test = cnn_read_data()


# 一个非常非常简陋的模型

# In[4]:


# Create the model
x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
y = tf.placeholder(tf.float32, [None, 22])


# ## 权重初始化
# 权重初始化时，将标准差从0.1调整为0.71

# In[5]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.71)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# # 卷积和池化

# In[6]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


print (x,y)



W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,6000,6000,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# # 第二层卷积

# In[9]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# # 密集连接层

# In[10]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# #  Dropout

# In[11]:


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# # 输出层

# In[12]:


W_fc2 = weight_variable([1024, 22])
b_fc2 = bias_variable([22])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# # 训练和评估模型

# ## 计算交叉熵

# In[13]:


cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) 
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[ ]:


init=tf.initialize_all_variables()
gpu_options = tf.GPUOptions(allow_growth=True,
                          visible_device_list="")
session_config = tf.ConfigProto(gpu_options=gpu_options)

sess = tf.Session(config=session_config)
sess.run(init)
for i in range(25000):
#    batch = mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy = accuracy.eval(session=sess,feed_dict={x:x_train, y: y_train, keep_prob: 0.5})
        print (type(train_accuracy))
        if train_accuracy>=0.14:
            print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(session=sess,feed_dict={x: x_train, y: y_train, keep_prob: 0.5})
            print ("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: x_train, y: y_train, keep_prob: 0.5}))
sess.close()



