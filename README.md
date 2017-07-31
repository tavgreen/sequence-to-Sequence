# Sequence to Sequence with Tensorflow #

## Introduction ##
In this tutorial, i would like to discuss how to implement Sequence-to-Sequence with Neural Network by[Sutskever et al](https://arxiv.org/abs/1409.3215). First of all, basic theory of Recurrent Neural Network (RNN) and its variant like Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU) will be discussed. The implementation consists of two parts: (1) implementation of LSTM in MNIST Dataset, (2) implementation of GRU in word prediction.

## Basic Theory ##
Deep Neural Networks is useful for learning tasks, but it has limitation: it can not used for mapping sequence-to-sequence. for instance: given a sequential word, we would like to predict next word based on previous word. another example: we would like to draw painting using neural networks, you don't have to paint all over again from scratch. you have to use previous draw painting from t step and complete drawing until resemble perfect painting. 
### 1. Recurrent Neural Network ###
Below is example of simple RNN graph (image from tensorflow):

![Fig.1](https://magenta.tensorflow.org/assets/rl_tuner/note_rnn_model.png "Tensorflow")

Green neuron is input layer in numeric format, yellow and red neuron are hidden layer and blue is output layer. Input for Deep learning usually in numeric format in order to easily compute learning parameter. how about word sequence? it should be changed into word vector representation read [Mikolov et al](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) paper. in RNN, hidden in time t (ht) is not only given from input (xt) but also from previous hidden (ht-1). it looks like we complete painting not from scratch, but from previous paintings. in above picture, let say input data xt(3) is "C", when forward into neural network, the label should be "G", the network will compute loss(how far "C" and "G" pairs) until obtain higher probability than others (let say "C" and "D" pairs or "C" and "A" pairs respectively). Learning can be used Stochastic Gradient Descent (SGD) with optimizer like ADAM or RMSProp. After computing loss, weight of network would be update until the end of epochs.

### 2. Long Short Term Memory (LSTM) ###
Simple RNN can cause **Vanishing Gradient** because hidden (ht) remember 1 or more previous hidden. One of solution is to use LSTM. LSTM resists of vanishing gradient, because it has forget layer that can forget previous hidden layer or reuse previous hidden layer. Example from [Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) can be useful:

![Fig.1](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png "Colah LSTM")

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
- [@Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [@Roatienza](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py)
- [@Aymericdarmien](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)

