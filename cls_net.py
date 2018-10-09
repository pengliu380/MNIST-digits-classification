# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:23:32 2018

@author: liupeng 
"""
# neural network for mnist digits classification
import random
import numpy as np
import matplotlib.pyplot as plt 

class Network(object):
    def __init__(self, sizes, activation = 'sigmoid'):
        '''
        sizes: the number of every layer's nodes
        sizes = [784, 30, 10],indicates 784 nodes in the input layer, 
        30 nodes in the hidden layer, and 10 nodes in the output layer.
        '''
        self.num_layers = len(sizes) # number of layers
        self.sizes = sizes # number of every layer's nodes
        # bias for every layer except the input layer
        self.bias = [0.01*np.ones((x, 1)) for x in sizes[1:]] 
        #self.bias = [np.random.randn(x,1)/np.sqrt(x) for x in sizes[1:]]
        # initialize the weights
        self.weights = [np.random.randn(y, x)/np.sqrt(y) for x,y in zip(sizes[:-1], sizes[1:])]
        #self.activation = activation
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.d_activation = d_sigmoid
        elif activation =='tanh':
            self.activation = tanh
            self.d_activation = d_tanh
        elif activation =='relu':
            self.activation = relu
            self.d_activation = d_relu
        
    def forward(self, x):
        # forward computation
        for w,b in zip(self.weights, self.bias):
            x = self.activation(np.dot(w, x) + b) #activation function: sigmoid function
        return x

    def train(self, train_data, epochs = 10, mini_batch_size = 10, lr = 0.01, test_data = None):
        '''
        train_data: 50,000 train data, (x,y):x is input, y is label
        epochs: Number of iterations, default is 30 times
        mini_batch_size: The size of the small batch of data at the time of sampling
        lr: learning rate
        test_data: if input the test data,test the data after every epoch
        '''
        train_data = list(train_data)
        n_train = len(train_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        train_errors = []
        test_accs = []
        # start train
        for i in range(epochs):
            # preprocess the data
            random.shuffle(train_data) # mess up the order
            mini_batches = [train_data[k: k+mini_batch_size]
                            for k in range(0, n_train, mini_batch_size)]  # divide batch
            # minibatch train
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr) 
            # train error
            
            train_error = 1.0 - float(self.evaluate(train_data))/n_train
            train_errors.append(train_error)
            print("Epoch: {} ,Train Error: {:.2%},".format(i + 1, train_error),end = '') 
            
            # test accuracy          
            if test_data:
                test_acc = float(self.evaluate(test_data))/n_test
                test_accs.append(test_acc)
                print("Test Accuracy: {:.2%}".format(test_acc))
        #plot
        plt.plot(train_errors)
        plt.title('training error')
        plt.xlabel('epoch')
        plt.show()
        plt.plot(test_accs)
        plt.title('test accuracy')
        plt.xlabel('epoch')
        plt.show()

    def update_mini_batch(self, mini_batch, lr):
        #initialize the delta w,b
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.bias]

        for x,y in mini_batch:
            y = vectorized_result(y)
            bp_w, bp_b = self.backprop(x,y)
            delta_w = [a+b for a,b in zip(delta_w, bp_w)]
            delta_b = [a+b for a,b in zip(delta_b, bp_b)] 
        #update weights and bias
        self.weights = [w-lr*dw for w,dw in zip(self.weights, delta_w)]           
        self.bias = [b-lr*db for b,db in zip(self.bias, delta_b)]
        
    def backprop(self, x, y):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.bias]
        activations = [x]
        for w,b in zip(self.weights, self.bias):
            x = self.activation(np.dot(w, x) + b)
            activations.append(x)
        delta = (activations[-1] - y) * self.d_activation(activations[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.d_activation(activations[-i])
            delta_b[-i] = delta
            delta_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (delta_w, delta_b)

    def evaluate(self,data):
        result = [(np.argmax(self.forward(x)), y) for (x,y) in data]
        return sum(int(x == y) for (x,y) in result)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))  

def d_sigmoid(x):
    #计算 σ函数的导数
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):  
    return np.maximum(0,x)

def d_relu(x):  
    return 1.0 * (x > 0)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    '''Derivative of the tanh function.'''
    return 1.0 - np.tanh(x)*np.tanh(x)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
if __name__ == '__main__':
    #load data
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    #train and test
    net = Network([784, 64, 32, 16, 10],activation = 'sigmoid')
    net.train(training_data, 50, 16, 1e-3, test_data=test_data)