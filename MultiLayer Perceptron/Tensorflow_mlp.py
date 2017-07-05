import tensorflow as tf
import numpy as np
from sklearn import datasets
def Onehotvector(d1):    
    train_target = []
    for ele in d1:
        if( ele == 0):
            train_target.append(np.array([1, 0, 0]))
        elif( ele == 1):
            train_target.append(np.array([0, 1, 0]))
        else:
            train_target.append(np.array([0, 0, 1]))
            
    return np.array(train_target)

x = tf.placeholder(tf.float32, [None, 4])
w1 = tf.Variable(tf.zeros([4, 3]))
b1 = tf.Variable(tf.zeros([3]))
y1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
             
w2 = tf.Variable(tf.zeros([3, 3]))
b2 = tf.Variable(tf.zeros([3]))
y = tf.matmul(y1, w2) + b2
             
d = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = d, logits = y))

train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

correct_predictions = tf.equal(tf.arg_max(y, 1), tf.arg_max(d, 1))

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

iris = datasets.load_iris()
train_data = np.array(iris.data[40:50, :])
d1 = np.array(iris.target[40:50])
train_data = np.concatenate((train_data, np.array(iris.data[90:100, :])), axis=0)
d1 = np.concatenate((d1, np.array(iris.target[90:100])), axis=0)
train_data = np.concatenate((train_data, np.array(iris.data[140:, :])), axis=0)
d1 = np.concatenate((d1, np.array(iris.target[140:])), axis=0)
train_target = Onehotvector(d1)

for i in range(10000):
    print("Epoch :{} Cross entropy: {}".format(i, sess.run(cross_entropy, {x: train_data, d: train_target})))
    sess.run(train_step, {x: train_data, d: train_target})

test_data = np.array(iris.data[:10, :])
d2 = np.array(iris.target[:10])
test_data = np.concatenate((test_data, np.array(iris.data[50:60, :])), axis=0)
d2 = np.concatenate((d2, np.array(iris.target[50:60])), axis=0)
test_data = np.concatenate((test_data, np.array(iris.data[100:110, :])), axis=0)
d2 = np.concatenate((d2, np.array(iris.target[100:110])), axis=0)
test_target = Onehotvector(d2)
print("\nTesting data\nPredicted: {} \nExpected:  {}".format(sess.run(tf.arg_max(y, 1), {x: test_data, d: test_target}), sess.run(tf.arg_max(d, 1), {d: test_target})))
print("Training accuracy: {}%".format(sess.run(accuracy*100, {x: train_data, d: train_target})))
print("Testing accuracy: {}%".format(sess.run(accuracy*100, {x: test_data, d: test_target})))

        
        
        
                                       
