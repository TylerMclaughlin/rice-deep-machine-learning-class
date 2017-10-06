__author__ = 'r_tyler_mclaughlin'
from three_layer_neural_network import *

import numpy as np

class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds a neural network with N hidden layers
    """

    def __init__(self, nn_input_dim, nn_depth, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units per hidden layer
        :param nn_hidden_dim: the number of hidden units per hidden layer
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        NeuralNetwork.__init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0)

        self.nn_depth = nn_depth

        # initialize the weights and biases in the network
        np.random.seed(seed)

        self.layers = []

        for l in range(0,nn_depth):
            if l == 0:
               self.layers.append(Layer(n_inputs = self.nn_input_dim, n_nodes = self.nn_hidden_dim))
            elif (l > 0) and (l<(nn_depth-1) ):
               self.layers.append(Layer(n_inputs = self.nn_hidden_dim, n_nodes = nn_hidden_dim))
            elif l == (nn_depth-1):
               self.layers.append(Layer(n_inputs = self.nn_hidden_dim, n_nodes = nn_output_dim,last_layer = True))

    def feedforward(self,X):
        for i,l in enumerate(self.layers):
            print(l)
            if i == 0:
               l.feedforward(X)
            else:
               l.feedforward(self.layers[i-1].a)
        self.probs = self.layers[i].probs

## RESUME HERE
    def backprop(self,y):
        for i in range(len(self.layers),-1,-1): # L,L-1,..., 1,0
            print(i)
            #self.layers[i].backprop()



class Layer(object):

    def __init__(self,n_inputs,n_nodes,actFun_type = 'tanh',last_layer = False):
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.initialize_weights()
        self.actFun_type = actFun_type
        self.last_layer = last_layer

    def __str__(self):
        return 'Inputs: %s, Nodes: %s' % (str(self.n_inputs), str(self.n_nodes))

    def initialize_weights(self):
        self.W = np.random.randn(self.n_inputs, self.n_nodes) / np.sqrt(self.n_nodes)
        self.b = np.zeros((1, self.n_nodes))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: tanh, sigmoid, or relu
        :return: activations
        '''
        if type == 'tanh':
            self.actFun_type = 'tanh'
            activation = np.tanh(z)
        elif type == 'sigmoid':
            self.actFun_type = 'sigmoid'
            # OLD CALCULATION
            # activation = 1. / (1 + np.exp(-z))
            # NEW CALCULATION using Scipy.special.expit
            activation = scipy.special.expit(z)
        elif type == 'relu':
            self.actFun_type = 'relu'
            return np.maximum(z, 0, z)
        else:
            print('%s is not a valid activation type' % (type))
            return None
        return activation

    def diff_actFun(self, act, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        AS AN EXPRESSION OF THE ACTIVATION
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        if type == 'tanh':
            self.actFun_type = 'tanh'
            # activation = 1 - np.actFun(z)**2.
            # deriv = 1 - np.tanh(z)**2.
            deriv = 1 - np.power(act, 2)
        elif type == 'sigmoid':
            self.actFun_type = 'sigmoid'
            deriv = act * (1 - act)
        elif type == 'relu':
            self.actFun_type = 'relu'
            deriv = act
            pos_indices = deriv > 0
            deriv[pos_indices] = 1.

        else:
            print('%s is not a valid activation type' % (type))
            return None
        return deriv


    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: layer input
        :param actFun: activation function
        :return:
        '''
        self.z = X.dot(self.W) + self.b
        if self.last_layer == True:
            # do softmax activation
            exp_scores = np.exp(self.z)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            # do normal activation
            self.a = self.actFun(self.z, type=self.actFun_type)

        return None

   ## RESUME HERE
    def backprop(self,y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: layer input
        :param y: layer output
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        num_examples = len(y)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        dW2 = (self.a1.T).dot(delta3) # dL/dW2
        db2 = np.sum(delta3, axis=0, keepdims=True)# dL/db2
        delta2 = delta3.dot(self.W2.T) * (self.diff_actFun(self.a1,type=self.actFun_type)) # (1 - np.power(self.a1,2))
        #delta2 = delta3.dot(self.W2.T) * self.diff_actFun(self.a1,type = self.actFun_type)
        dW1 = np.dot(X.T,delta2) #dL/dW1
        db1 = np.sum(delta2,axis = 0) #dL/db1
        return dW1, dW2, db1, db2



def main():
  # generate and visualize Make-Moons dataset
  X, y = generate_data()
  # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
  # plt.show()

  model = DeepNeuralNetwork(nn_input_dim=2, nn_depth = 3, nn_hidden_dim=3, nn_output_dim=2, actFun_type='sigmoid')
  #model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='sigmoid')
  model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()