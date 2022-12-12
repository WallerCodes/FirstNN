import math
import numpy as np


############################################################################################################################################

# In this program, we will create a class that builds a neural network from scratch and use it to identify the images of digits 0-9 imported
# from the sklearn library.

############################################################################################################################################


# Create the ‘Nnetwork’ class and define its arguments:
# Set the number of neurons/nodes for each layer
# and initialize the weight matrices:


class Nnetwork:

    def __init__(self):
        # Initialize the values of the NN to the parameters passed into the function
        self.no_of_in_nodes = 8 * 8  # Too lazy to customize rn so hardcoded
        self.no_of_out_nodes = 10  # Right now we are doing 10 because we want to be able to recognize 10 different digits 0-9
        self.no_of_hidden_nodes = math.floor(
            self.no_of_in_nodes * 2 / 3) + self.no_of_out_nodes  # Formula for calculating appropriate amount of hidden nodes
        self.no_of_hidden_layers = 2  # This number will help determining how many nodes per hidden layer there are
        self.learning_rate = .01  # How strongly the weights will change after back propagation

        # Calculate the number of nodes per hidden layer to make life easier.
        # We are assuming that all hidden layers have the same amount of nodes
        self.no_of_nodes_per_hidden_layer = int(self.no_of_hidden_nodes / self.no_of_hidden_layers)
        # print("Number of nodes per hidden layer:", self.no_of_nodes_per_hidden_layer)

        # Initialize the weight matrices of the NN
        # We have 2 weight matrices, one for the hidden layers and one for the output layer because they all differ in size
        self.first_hidden_layer_weights = np.random.rand(self.no_of_in_nodes + 1,
                                                         self.no_of_nodes_per_hidden_layer)  # The +1 here is for the bias weights.
        # The first row of the weight matrix is the bias weights and the following rows are the weights of the input neurons
        # This means that technically every single neuron in the hidden layer will have one bias node connected to it multiplied by a weight
        # But in reality it is just one bias node with a weight going into every single node of the hidden layer. Each of these weights
        # counts as one parameter. E.g. if the input layer had 5 neurons and a bias node, and the output layer has 4 neurons, then the model
        # has 4*(5+1) parameters. 5 weights plus the bias weight for each node, and that for each of the 4 nodes in the output layer.
        # print(self.hidden_layer_weights.shape)

        self.second_hidden_layer_weights = np.random.rand(self.no_of_nodes_per_hidden_layer + 1,
                                                          self.no_of_nodes_per_hidden_layer)

        self.output_layer_weights = np.random.rand(self.no_of_nodes_per_hidden_layer + 1,
                                                   self.no_of_out_nodes)  # The + 1 here is for the bias weights again

        # print(self.output_layer_weights.shape)

        # Initialize node matrices of the NN. These will hold the activation values of the nodes and are initialized to 0
        self.input_layer_nodes = np.zeros((self.no_of_in_nodes + 1, 1))  # + 1 for extra bias node
        self.input_layer_nodes[0, 0] = 1
        # print(self.input_layer_nodes)
        self.first_hidden_layer_nodes = np.zeros(
            (self.no_of_nodes_per_hidden_layer + 1, 1))  # +1 for extra bias node
        self.first_hidden_layer_nodes[0, 0] = 1

        self.second_hidden_layer_nodes = np.zeros(
            (self.no_of_nodes_per_hidden_layer + 1, 1))  # +1 for extra bias node
        self.second_hidden_layer_nodes[0, 0] = 1

        self.output_layer_nodes = np.zeros((self.no_of_out_nodes, 1))

        # print(self.input_layer_nodes.shape)
        # print(self.hidden_layer_nodes.shape)
        # print(self.output_layer_nodes.shape)

    ########################################################################################################################################

    # This will be out method for training our neural network. It takes an input vector and the desired output (labels)
    def forwardprop(self, input_vector, target_vector):
        # First we need to normalize the input vector, otherwise the tanh function will just return 0s and 1s
        # print(input_vector.shape)

        norm_input = input_vector / np.linalg.norm(input_vector)
        # Let's assume that the input we receive is already in the vector form we need for simplicity.
        self.input_layer_nodes[1:, :] = norm_input  # Initialize the input layer nodes to the input.
        # Index every node starting from 1 instead of 0 since the 0th node is the bias node
        # print(self.input_layer_nodes)

        # ------------------------

        # Now we need to apply all the weights of the input layer to the first hidden layer
        # The way the weights are organized each, row represents all the weights for a single neuron in the input layer. If we take the
        # transpose of this matrix, then each column will represent all the weights going from that specific node of the input layer to the
        # nodes of the next layer (hidden layer).
        # In this case we only have two hidden layers, so I did it by hand, but in the case of many hidden layers we need to loop through
        # them here.
        # We will perform the matrix multiplication, but then we need to normalize the data so that the tanh function can make something
        # useful out of it.

        # ----------------------------
        # Forward propagation from input layer into first hidden layer
        first_hidden_layer_nodes = np.matmul(self.first_hidden_layer_weights.transpose(), self.input_layer_nodes)
        # print(hidden_layer_nodes)

        # This gives us the weighted sum of the activation from the input layer
        # print(hidden_layer_nodes)
        norm_first_hidden_layer_nodes = first_hidden_layer_nodes / np.linalg.norm(first_hidden_layer_nodes)
        # print(norm_hidden_layer_nodes)
        self.first_hidden_layer_nodes[1:, :] = tanh(norm_first_hidden_layer_nodes)  # Larger number give all 1's
        # print(self.hidden_layer_nodes)

        # ----------------------------
        # Forward propagation from first hidden layer into second hidden layer
        second_hidden_layer_nodes = np.matmul(self.second_hidden_layer_weights.transpose(), self.first_hidden_layer_nodes)
        # print(hidden_layer_nodes)

        # This gives us the weighted sum of the activation from the input layer
        # print(hidden_layer_nodes)
        norm_second_hidden_layer_nodes = second_hidden_layer_nodes / np.linalg.norm(second_hidden_layer_nodes)
        # print(norm_hidden_layer_nodes)
        self.second_hidden_layer_nodes[1:, :] = tanh(norm_second_hidden_layer_nodes)  # Larger number give all 1's

        # ----------------------------
        # Forward propagation from second hidden layer into output layer
        output_layer_nodes = np.matmul(self.output_layer_weights.transpose(), self.second_hidden_layer_nodes)
        norm_output_layer_nodes = output_layer_nodes / np.linalg.norm(output_layer_nodes)
        self.output_layer_nodes = tanh(norm_output_layer_nodes)
        # print(self.output_layer_nodes)

        # print('Loss:', self.loss(target_vector))

    ########################################################################################################################################

    def backprop(self, target, learning_rate):
        """All of these steps are explained in detail in this video: https://www.youtube.com/watch?v=tIeHLnjs5U8"""
        # Backprop would usually contain a loop but since we only have 2 hidden layers we can do it manually for clarity

        # -------------------------------------------------------------------------------------------------------------------------------- #
        # WEIGHT ADJUSTMENT OUTPUT LAYER
        # z is the derivative of the activation of the previous layer multiplied by the weights with the bias added on
        # It is needed for calculating the derivatives of the components of the derivative of the cost function with respect to the weights
        z = np.matmul(self.output_layer_weights.transpose(),
                      self.second_hidden_layer_nodes)

        # print(z.shape)

        # Again normalize for tanh function
        z_norm = z / np.linalg.norm(z)

        # This is the derivative of z with respect to the weights
        hidden_activation = self.second_hidden_layer_nodes  # Doesn't need to be normalized because it already went through norm tanh
        # print(hidden_activation)

        # This is the derivative of the tanh function
        tanh_derivative = 1 - np.square(tanh(z_norm))

        # This is the derivative of the cost with respect to the activation of the current layer
        cost_derivative = 2 * (self.output_layer_nodes - target)

        # The product of all three of these combined should
        # give us the negative gradient meaning the amount which we should adjust the
        # weights to move towards our desired result

        output_layer_gradient = -np.matmul(hidden_activation, np.multiply(tanh_derivative, cost_derivative).transpose())[1:, :]
        # print(output_layer_gradient)

        # After this, we need to add the result to our weight matrix so that the adjustments take effect
        # Do not modify these weights yet as we need them to calculate the activation of the previous layer that minimizes the cost function

        # BIAS WEIGHT ADJUSTMENT OUTPUT LAYER

        bias_weights_output_layer = np.multiply(tanh_derivative, cost_derivative).transpose()

        # -------------------------------------------------------------------------------------------------------------------------------- #
        # WEIGHT ADJUSTMENT SECOND HIDDEN LAYER
        ideal_activation = np.matmul(self.output_layer_weights[1:, :], np.multiply(tanh_derivative, cost_derivative))
        norm_ideal_activation = ideal_activation / np.linalg.norm(ideal_activation)

        z = np.matmul(self.second_hidden_layer_weights.transpose(),
                      self.first_hidden_layer_nodes)  # 0 is our bias here, but we will have a matrix for it in the future

        # Again normalize for tanh function
        z_norm = z / np.linalg.norm(z)

        # This is the derivative of z with respect to the weights
        input_activation = self.first_hidden_layer_nodes

        # print(input_activation)

        # This is the derivative of the sigmoid function
        # sigmoid_derivative = np.exp(-z_norm) / np.square(1 + np.exp(-z_norm))
        # print(sigmoid_derivative.shape)

        # This is the derivative of tanh
        tanh_derivative = 1 - np.square(tanh(z_norm))

        # This is the derivative of the cost with respect to the output layer of this example
        cost_derivative = 2 * (
                self.second_hidden_layer_nodes[1:, :] - norm_ideal_activation)  # We can ignore the bias activation in this case
        # because there is no weight going from the input layer into the bias node
        # print(cost_derivative.shape)

        # The product of all three of these combined should give us the negative gradient meaning the amount which we should adjust the
        # weights to move towards our desired result
        # We want to omit the first row because we will be calculating the change of weights for the bias separately
        second_hidden_layer_gradient = -np.matmul(input_activation, np.multiply(tanh_derivative, cost_derivative).transpose())[1:, :]
        # print(second_hidden_layer_gradient.shape)

        # BIAS WEIGHT ADJUSTMENT SECOND HIDDEN LAYER

        bias_weights_second_hidden_layer = np.multiply(tanh_derivative, cost_derivative).transpose()

        # -------------------------------------------------------------------------------------------------------------------------------- #
        # Here we will calculate the desired activation of the previous layer to minimize the cost function

        ideal_activation = np.matmul(self.second_hidden_layer_weights, np.multiply(tanh_derivative, cost_derivative))
        norm_ideal_activation = ideal_activation / np.linalg.norm(ideal_activation)

        # WEIGHT ADJUSTMENT FIRST HIDDEN
        # Then we need to do the same for the weights between the first hidden layer and the input layer
        # We need to get the derivative of the cost function with respect to the activation of the previous layer so that we can calculate
        # the difference of the activation of that layer and the activation of that layer that would minimize the cost.

        z = np.matmul(self.first_hidden_layer_weights.transpose(),
                      self.input_layer_nodes)  # 0 is out bias here, but we will have a matrix for it in the future

        # Again normalize for tanh function
        z_norm = z / np.linalg.norm(z)

        # This is the derivative of z with respect to the weights
        input_activation = self.input_layer_nodes

        # print(input_activation)

        # This is the derivative of the sigmoid function
        # sigmoid_derivative = np.exp(-z_norm) / np.square(1 + np.exp(-z_norm))
        # print(sigmoid_derivative.shape)

        # This is the derivative of tanh
        tanh_derivative = 1 - np.square(tanh(z_norm))

        # This is the derivative of the cost with respect to the output layer of this example
        cost_derivative = 2 * (
                self.first_hidden_layer_nodes[1:, :] - norm_ideal_activation[1:, :])  # We can ignore the bias activation in this case
        # because there is no weight going from the input layer into the bias node
        # print(cost_derivative.shape)

        # The product of all three of these combined should give us the negative gradient meaning the amount which we should adjust the
        # weights to move towards our desired result
        first_hidden_layer_gradient = -np.matmul(input_activation, np.multiply(tanh_derivative, cost_derivative).transpose())[1:, :]
        # print(hidden_layer_gradient)

        # BIAS WEIGHT ADJUSTMENT OUTPUT LAYER

        bias_weights_first_hidden_layer = np.multiply(tanh_derivative, cost_derivative).transpose()

        # print(self.first_hidden_layer_weights[0])

        # -------------------------------------------------------------------------------------------------------------------------------- #

        # Now we can update all weights with their respective new gradients
        self.output_layer_weights[1:, :] = self.output_layer_weights[1:, :] + output_layer_gradient * learning_rate
        self.output_layer_weights[0] = bias_weights_output_layer * learning_rate

        self.second_hidden_layer_weights[1:, :] = self.second_hidden_layer_nodes[1:, :] + second_hidden_layer_gradient * learning_rate
        self.second_hidden_layer_weights[0] = bias_weights_second_hidden_layer * learning_rate

        self.first_hidden_layer_weights[1:, :] = self.first_hidden_layer_weights[1:, :] + first_hidden_layer_gradient * learning_rate
        self.first_hidden_layer_weights[0] = bias_weights_first_hidden_layer * learning_rate

    ########################################################################################################################################

    def predict(self, input_vector, target_vector):
        """
        running the network with an input vector 'input_vector'.
        'input_vector' can be tuple, list or ndarray
        """

        self.forwardprop(input_vector=input_vector, target_vector=target_vector)
        return self.output_layer_nodes.argmax(axis=0)

    ########################################################################################################################################

    # This is not mean squared error but just squared error
    def loss(self, target_vector):
        return sum(np.square(self.output_layer_nodes - target_vector))

    ########################################################################################################################################


############################################################################################################################################

def tanh(array):
    return (np.exp(array) - np.exp(-array)) / (np.exp(array) + np.exp(-array))


############################################################################################################################################

def xavierWeightInitialization(array):
    # number of nodes in the previous layer
    n = array.shape[0]
    # calculate the range for the weights
    lower, upper = -(1.0 / math.sqrt(n)), (1.0 / math.sqrt(n))
    array = lower + array * (upper - lower)

############################################################################################################################################
