import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing, cross_validation
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2

attributes=pd.read_csv('cancer_attributes.txt')
classes=pd.read_csv('cancer_classes.txt')

attributes=preprocessing.scale(attributes)

x_train,x_test,y_train,y_test=cross_validation.train_test_split(attributes,classes,test_size=0.2)

x = tf.placeholder('float', [None, 9])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID')

def convolutional_neural_network(x):
	weights = {'W_conv1':tf.get_variable('w_conv1',shape=[2,2,1,32], initializer=tf.contrib.layers.xavier_initializer()),
			   'W_conv2':tf.get_variable('w_conv2',shape=[2,2,32,64],initializer=tf.contrib.layers.xavier_initializer()),
			   'W_fc':tf.get_variable('w_fc',shape=[1*1*64,1024],initializer=tf.contrib.layers.xavier_initializer()),
			   'out':tf.get_variable('wout',shape=[1024,n_classes],initializer=tf.contrib.layers.xavier_initializer())}
	biases = {'b_conv1':tf.get_variable('B_conv1',shape=[32], initializer=tf.contrib.layers.xavier_initializer()),
				'b_conv2':tf.get_variable('B_conv2',shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
				'b_fc':tf.get_variable('B_fc',shape=[1024], initializer=tf.contrib.layers.xavier_initializer()),
				'out':tf.get_variable('Bout',shape=[n_classes], initializer=tf.contrib.layers.xavier_initializer())}
    
	x = tf.reshape(x, shape=[-1, 3, 3, 1])
	print(x)
	tf.summary.histogram("weights",weights['W_conv1'])
	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)
    
 	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
  	conv2 = maxpool2d(conv2)
 	print(conv2)
  	fc = tf.reshape(conv2,[-1, 1*1*64])
  	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
  	fc = tf.nn.dropout(fc, keep_rate)

 	output = tf.matmul(fc, weights['out'])+biases['out']

  	return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ,name='cost')
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    tf.summary.scalar('cost',cost)
    #Scalar
    
    
    
    hm_epochs = 200
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        writer2=tf.summary.FileWriter("/home/samarth/BreastCancerConvNet/output4")
        writer2.add_graph(sess.graph)
        merged_summary=tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0;epoch_x=x_train;epoch_y=y_train
            
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            s=sess.run(merged_summary,feed_dict={x: epoch_x, y: epoch_y})
            writer2.add_summary(s,epoch)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'),name='accuracy')
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))
        
    #The graphs saver
    #writer=tf.summary.FileWriter("/home/samarth/BreastCancerConvNet/output")
    #writer.add_graph(sess.graph)
    
    # Command to run - sudo tensorboard --logdir /home/samarth/BreastCancerConvNet/output
    #All sumaries
    
    
train_neural_network(x)
