# Sequence to Sequence with Tensorflow #

## Introduction ##
In this tutorial, i would like to discuss how to implement Sequence-to-Sequence with Neural Network by[Sutskever et al](https://arxiv.org/abs/1409.3215). First of all, basic theory of Recurrent Neural Network (RNN) and its variant like Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU) will be discussed. The implementation consists of two parts: (1) implementation of LSTM in MNIST Dataset, (2) implementation of GRU in word prediction.

## Basic Theory ##
Deep Neural Networks is useful for learning tasks, but it has limitation: it can not used for mapping sequence-to-sequence. for instance: given a sequential word, we would like to predict next word based on previous word. another example: we would like to draw painting using neural networks, you don't have to paint all over again from scratch. you have to use previous draw painting from t step and complete drawing until resemble perfect painting.

** Not Completed yet **

## Implementation ##
### 1. MNIST Prediction using LSTM ##
MNIST dataset contains 50000 training digit images and 10000 testing digit images. For each image 28x28 pixels can be changed into sequence tasks.
- import libraries and set parameter for learning
```python
import tensorflow as tf
from tensorflow.contrib import rnn as tfrnn
lr = 0.001
epochs = 50000
displaystep = 100
n_hidden = 128
n_class = 10
n_input = 28
```
- create class LSTM which contains graph.
```python
def lstm(x, weights, biases):
  #unstack input into 28 input size
  x = tf.unstack(x, n_input, 1)  
  #create lstm cell that connect state and output from previous lstm cell(see.fig 1)
  rnn_cell = tfrnn.MultiRNNCell([tfrnn.BasicLSTMCell(n_hidden), tfrnn.BasicLSTMCell(n_hidden)]) 
  #create output layer and states from input and lstm cell
  outputs, states = tfrnn.static_rnn(rnn_cell, x, dtype=tf.float32)
  #create output layer( Wt * outputLSTMt + biast)
  return tf.add(tf.matmul(outputs[-1], weights['out']),biases['out'])

```
- provide placeholder for input and output and definition of weights and biases for output layer
```python
x = tf.placeholder(tf.float32, [None, n_input, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
weights = {
  'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	'out': tf.Variable(tf.random_normal([n_classes]))
}
```
- Setting Graph
```python
y_pred = lstm(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
```
- Run Training
```python
with tf.Session() as sess:
	step = 0;acc_total = 0;loss_total = 0
	while step * batchsize < epochs:
		batch_x, batch_y = mnist.train.next_batch(batchsize)
		batch_x = batch_x.reshape((batchsize, n_input, n_input))
		sess.run(optimizer,feed_dict={x: batch_x, y: batch_y})
		step += 1
```
- Run Testing
```python
x_test = mnist.test.images.reshape((-1, n_input, n_input))
y_test = mnist.test.labels
print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x:x_test, y: y_test}))
```

### 2. Word Prediction using GRU ###
** Not Finished Yet **

## Result ##
MNIST Prediction using LSTM produces 93% accuracy in testing. Loss and Accuracy in training can be seen in below picture:
![Fig.1](https://raw.github.com/tavgreen/sequence-to-sequence/master/file/lstm1.png?raw=true "result")

## References ##
- Sutskever et al. 2014. Sequence to Sequence Learning with Neural Network. https://arxiv.org/abs/1409.3215 
- [@Roatienza](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py)
- [@Aymericdarmien](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)

