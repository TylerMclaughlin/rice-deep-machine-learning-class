import three_layer_neural_network

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
               self.layers.append(Layer(n_inputs = self.nn_hidden_dim, n_nodes = nn_output_dim))



class Layer(object):
    def __init__(self,n_inputs,n_nodes,actFun_type = 'tanh',last_layer = False):
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.initialize_weights()
        self.actFun_type = actFun_type
        self.last_layer = last_layer

    def initialize_weights(self):
        self.W = np.random.randn(self.n_inputs, self.n_nodes) / np.sqrt(self.n_nodes)
        self.b = np.zeros((1, self.n_nodes))

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
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

    def __str__(self):
       return 'Inputs: %s, Nodes: %s' % (str(self.n_inputs), str(self.n_nodes))

def main():
  # generate and visualize Make-Moons dataset
  X, y = generate_data()
  # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
  # plt.show()

  #model = DeepNeuralNetwork(nn_input_dim=2, nn_depth = 3, nn_hidden_dim=3, nn_output_dim=2, actFun_type='tanh')
  model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='sigmoid')
  model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()