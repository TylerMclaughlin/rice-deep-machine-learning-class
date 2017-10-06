__author__ = 'r_tyler_mclaughlin'

# tremendous credit goes to the following sources:
# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
#https://github.com/abhmul/COMP-576/blob/master/Assignment1/n_layer_network.py


from three_layer_neural_network import generate_data, generate_data_wine
from three_layer_neural_network import generate_data_bc, generate_data_blobs, NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt
import scipy.special

ACT_FUNS = { 'sigmoid': scipy.special.expit,'tanh': np.tanh,
               'relu': lambda x: np.maximum(x, 0, x)}

DIFF_ACT_FUNS = { 'sigmoid': lambda x: scipy.special.expit(x) * scipy.special.expit(1. - x),
                'tanh': lambda x: 1 - np.square(np.tanh(x)),
                'relu': lambda x: (x > 0).astype(float)}
#epsilon to avoid divide-by-zero
EPS = 1e-14

class DeepNeuralNetwork(NeuralNetwork):

    def __init__(self, layers, seed=0):
        '''
        :param layers: the layers in sequential order of the neural network
        :param seed: random seed
        '''
        self.layers = layers
        self.input_dim = self.layers[0].n_inputs
        self.output_dim = self.layers[-1].n_nodes
        self.probs = None

    def actFun(self, z, type):
         '''
         actFun computes the activation functions
         :param z: net input
         :param type: tanh, sigmoid, or relu
         :return: activations
         '''
         return ACT_FUNS[type](z)

    def feedforward(self, X):
        assert self.input_dim == X.shape[1], "Net input size is %s but passed array with %s features" % (
            self.input_dim, X.shape[1])
        tensor = X
        for layer in self.layers:
            tensor = layer(tensor)
        self.probs = tensor
        return tensor

    def __call__(self, X):
        return self.feedforward(X)

    def backprop(self, y):
        delta_term = y
        for layer in reversed(self.layers):
            delta_term = layer.backprop(delta_term)

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels - Shape (samples,)
        :return: the loss for prediction
        '''
        num_examples = len(X)
        # Forward propagation
        self(X)
        data_loss = np.sum(-np.log(self.probs[np.arange(num_examples), y]))
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            print(i)
            # Forward propagation
            self(X)
            # Backpropagation
            self.backprop(y)

            # Gradient descent parameter update
            for layer in self.layers:
                layer.update(epsilon)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" %
                      (i, self.calculate_loss(X, y)))

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self(X)
        return np.argmax(self.probs, axis=1)

class Layer(object):

    def __init__(self,n_inputs,n_nodes,actFun_type = 'tanh',reg_lambda = 0.01,seed=0,last_layer=False):
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.actFun_type = actFun_type
        self.last_layer = last_layer
        self.reg_lambda = reg_lambda

        self.delta = None
        self.x = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None

        # Initialize weights
        np.random.seed(seed)
        self.W = np.random.randn(self.n_inputs, self.n_nodes) / np.sqrt(self.n_nodes)
        self.b = np.zeros((1, self.n_nodes))

        if last_layer == True:
            self.probs = None

    def __str__(self):
        return 'Inputs: %s, Nodes: %s' % (str(self.n_inputs), str(self.n_nodes))


    def actFun(self, z):
        '''
        actFun computes the activation functions
        :param z: net input
        :return: activations
        '''

        return ACT_FUNS[self.actFun_type](z)

    def diff_actFun(self, z):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        AS AN EXPRESSION OF THE ACTIVATION

        :return: the derivatives of the activation functions wrt the net input
        '''

        return DIFF_ACT_FUNS[self.actFun_type](z)

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: layer input
        :param actFun: activation function
        :return:
        '''
        self.x = X
        self.z = X.dot(self.W) + self.b
        if self.last_layer == True:
            # do softmax activation
            exp_scores = np.exp(self.z)
            self.a = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + EPS)
        else:
            # do normal activation
            self.a = self.actFun(self.z)
            return self.a


    def backprop(self,y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: layer input
        :param y: layer output
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''
        if self.last_layer == False:
            da = self.diff_actFun(self.z)
            delta = y * da
            self.dW = np.dot(self.x.T, delta)
            self.db = np.sum(delta, axis=0)

        else: # if output layer
            num_examples = len(self.x)
            delta = self.a
            delta[range(num_examples), y] -= 1

            dregW = (self.reg_lambda * self.W)
            self.dW = np.dot(self.x.T, delta) + dregW
            self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)

    def __call__(self, x):
        return self.feedforward(x)

    def update(self, epsilon):
        self.W += -epsilon * self.dW
        self.b += -epsilon * self.db


class OutputLayer(Layer):

    def __init__(self, input_size, output_size, reg_lambda=0.01, seed=0):
        super(OutputLayer, self).__init__(input_size, output_size, 'linear',
                                           reg_lambda, seed)
        self.probs = None

    def actFun(self, z):
        '''
        actFun computes the sofmtax activation functions
        :param z: net input
        :return: activations
        '''
        exp_scores = np.exp(z - np.max(z))
        return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + EPS)

    def diff_actFun(self, z):
        '''
        diff_actFun computes the derivatives of the softmax functions wrt the layer input
        :param z: net input
        :return: the derivatives of the softmax functions wrt the net input
        '''

        raise ValueError("Softmax is only meant to be used as final layer")

    def backprop(self, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param x: input data
        :param y: given labels - Shape (samples,)
        '''

        num_examples = len(self.x)
        delta = self.a
        delta[range(num_examples), y] -= 1

        dregW = (self.reg_lambda * self.W)
        self.dW = np.dot(self.x.T, delta) + dregW
        self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)


def main():
  # generate and visualize Make-Moons dataset
  X, y = generate_data()
  # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
  # plt.show()
  # generate and visualize Load Wine Dataset
  # doesn't work because of too many features for plot_decision_boundary.
  #X, y = generate_data_wine()
  # doesn't work  because ditto
  #X, y = generate_data_bc()
  #X, y = generate_data_blobs(centers=3)
  #X, y = generate_data_blobs(centers=4)
  #X, y = generate_data_blobs(centers=6)
  #X, y = generate_data_blobs(centers=12)

  #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Set1)
  # plt.show()

  #layer_sizes = [X.shape[1], 10, 6, 2, 3, 3]

  #layer_sizes = [X.shape[1], 3, 3, 3, 3]

  # brilliant for make moons!
  layer_sizes = [X.shape[1], 6, 6]
  #layer_sizes = [X.shape[1], 3, 3]
  #layer_sizes = [X.shape[1], 10, 8, 6, 4, 3]
  #layer_sizes = [X.shape[1], 3, 4, 6, 8, 10]

  # try for wines
  #layer_sizes = [X.shape[1],6,6]# 16, 16]

  #layer_sizes = [X.shape[1], 2, 4, 10, 4, 3]
  #layer_sizes = [X.shape[1], 3]
  layers = [Layer(n_inputs = layer_sizes[i], n_nodes = layer_sizes[i + 1], actFun_type ='sigmoid')
            for i in range(len(layer_sizes) - 1)]
  #layers.append(OutputLayer(input_size = layer_sizes[-1], output_size = 2))
  layers.append(OutputLayer(input_size = layer_sizes[-1], output_size = max(y) + 1))

  #model = DeepNeuralNetwork(nn_input_dim=2, nn_depth = 3, nn_hidden_dim=3, nn_output_dim=2, actFun_type='sigmoid')
  model = DeepNeuralNetwork(layers)
  #model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='sigmoid')
  #model.fit_model(X, y, num_passes=2000)
  model.fit_model(X, y, epsilon =0.01, num_passes=20000)
  model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()