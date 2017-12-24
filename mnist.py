import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

#Creating variables

def weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

#Convolution layer structure

def convolutional_layer(x,filter_size,num_filters,color_channels):
	filter_shape=[filter_size,filter_size,color_channels,num_filters]
	W=weights(filter_shape)
	b=biases(num_filters)
	new_layer=tf.nn.conv2d(x,W,[1,1,1,1],"SAME")
	new_layer+=b
	new_layer=tf.nn.max_pool(new_layer,[1,2,2,1],[1,2,2,1],"SAME")
	new_layer=tf.nn.relu(new_layer)

	return new_layer

#Flatten layer to feed to FC

def flat(layer):
	shape=layer.get_shape().as_list()
	features=shape[1]*shape[2]*shape[3]
	layer=tf.reshape(layer,[-1,features])

	return layer,features

def FC(x,inputs,outputs):
	W=weights(shape=[inputs,outputs])
	b=biases(length=outputs)
	layer = tf.matmul(x, W) + b

	return layer

def plot_image(image,i):
	image=np.reshape(image,(28,28))
	plt.imshow(image,cmap='gray')
	plt.title('Perturbation:'+str(i))
	plt.show()
	
	
	


 #Construction of graph

x = tf.placeholder(tf.float32,[None, 28*28], 'x')
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_image=tf.placeholder(tf.float32,[None,10], 'y_image')
y=tf.argmax(y_image, axis=1)
e=tf.placeholder(dtype=tf.float32,name='e')

layer_one=convolutional_layer(x_image,5,4,1)
layer_two=convolutional_layer(layer_one,5,16,4)
layer_three=convolutional_layer(layer_two,5,36,16)

flattened_layer,features=flat(layer_three)
FC_one=FC(flattened_layer,features,120)
FC_one=tf.nn.relu(FC_one)
FC_two=FC(FC_one,120,10)

#Output of layer

y_pred_image=tf.nn.softmax(FC_two)
y_pred=tf.argmax(y_pred_image,axis=1)

#Cost function and optimisation

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=FC_two,labels=y_image)
cost = tf.reduce_mean(cross_entropy)

#adversial image
grad=tf.gradients(cost,x_image)
x_adv=tf.sign(grad)*e+x_image
x_adv=tf.clip_by_value(x_adv,0,1)
x_adv=tf.reshape(x_adv,[-1,28*28])
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
prediction=tf.equal(y,y_pred)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))



#Running of session
batch_size=50
test_x=data.test.images
test_y=data.test.labels
feed_dict_test = {x: test_x,y_image: test_y}

with tf.Session() as session:
  	session.run(tf.global_variables_initializer())
  	print("Training model...\n")
  	for i in range(0,10001):
		x_batch,y_batch=data.train.next_batch(batch_size)
		feed_dict_train = {x: x_batch,y_image: y_batch}
		session.run(optimizer, feed_dict=feed_dict_train)
		if i % 100 == 0:
			acc = session.run(accuracy, feed_dict=feed_dict_train)
			acc_test = session.run(accuracy,feed_dict=feed_dict_test)
			print('Iteration : %d Train Accuracy : %g Test Accuracy : %g\n' % (i,acc,acc_test))
	
	numbers=[0.1,0.15,0.2,0.25,0.3,0.35]
	print("Testing with adversial images...\n")
	plot_image(test_x[1],0)
	for i in numbers:
  		x_adversial=session.run(x_adv,feed_dict={x: test_x,y_image: test_y,e: i})
  		acc_adv=session.run(accuracy,feed_dict={x: x_adversial,y_image:test_y})
  		print('Test accuracy with only adversial images : %g with perturbation : %g' % (acc_adv,i))
		plot_image(x_adversial[1],i)

		

		



		

