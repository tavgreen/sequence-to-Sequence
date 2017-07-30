import tensorflow as tf
from tensorflow.contrib import rnn as tfrnn
from tensorflow.examples.tutorials.mnist import input_data
'''
Iter= 0, Average Loss= 2.670063, Average Accuracy= 9.38%
Iter= 12800, Average Loss= 1.042971, Average Accuracy= 71.09%
Iter= 25600, Average Loss= 0.315112, Average Accuracy= 89.06%
Iter= 38400, Average Loss= 0.173301, Average Accuracy= 95.31%
Testing Accuracy:  0.9356
'''
class RNN:
	def __init__(self, lr, epochs, batchsize, displaystep, dataset ):
		self.lr = lr
		self.epochs = epochs
		self.batchsize = batchsize
		self.displaystep = displaystep
		self.dataset = dataset
		self.n_hidden = 128
		self.n_classes = 10
		self.n_input = 28

	def lstm(self, x, weights, biases):
		x = tf.unstack(x, self.n_input, 1)
		#rnn_cell = tfrnn.BasicLSTMCell(self.n_hidden)
		rnn_cell = tfrnn.MultiRNNCell([tfrnn.BasicLSTMCell(self.n_hidden), tfrnn.BasicLSTMCell(self.n_hidden)]) #rnn_cell derived from previous rnn
		outputs, states = tfrnn.static_rnn(rnn_cell, x, dtype=tf.float32) #produce y output and states from rnn_cell
		return tf.add(tf.matmul(outputs[-1], weights['out']),biases['out']) #connect weights and output from rnn cell

	def training(self):
		x = tf.placeholder(tf.float32, [None, self.n_input, self.n_input])
		y = tf.placeholder(tf.float32, [None, self.n_classes])
		weights = {
			'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
		}
		biases = {
			'out': tf.Variable(tf.random_normal([self.n_classes]))
		}
		y_pred = self.lstm(x, weights, biases)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
		optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(cost)
		correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1)) #correct: if pred = y
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #accuracy
		init = tf.global_variables_initializer() #global var initialization
		with tf.Session() as sess: #running graph
			sess.run(init) #run variable
			step = 0;acc_total = 0;loss_total = 0
			while step * self.batchsize < self.epochs:
				batch_x, batch_y = mnist.train.next_batch(self.batchsize)
				batch_x = batch_x.reshape((self.batchsize, self.n_input, self.n_input))
				sess.run(optimizer,feed_dict={x: batch_x, y: batch_y}) #feed forward
				#loss_total += loss #count loss
				#acc_total += acc #count accuracy
				if step % self.displaystep == 0: #if equals display step then show
					acc_total,loss_total = sess.run([accuracy,cost], feed_dict={x: batch_x, y: batch_y})
					print("Iter= " + str(step * self.batchsize) + ", Average Loss= {:.6f}".format(loss_total) + ", Average Accuracy= {:.2f}%".format(100*acc_total))
				step += 1
			print("Finish")
			x_test = mnist.test.images.reshape((-1, self.n_input, self.n_input))
			y_test = mnist.test.labels
			print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x:x_test, y: y_test}))


LEARNING_RATE = 0.001
EPOCHS = 50000
BATCH_SIZE = 128
DISPLAY_STEP = 100
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

rnn = RNN(LEARNING_RATE, EPOCHS, BATCH_SIZE, DISPLAY_STEP,mnist)
rnn.training()