__author__ = 'tan_nguyen'

import numpy as np
import torch.nn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_data():
    """
    generate data
    :return: X: input data, y: given labels
    """
    X, y = datasets.load_wine(return_X_y=True)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    """
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSIGNMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        """
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        """
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        """
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        """
        if type == "tanh":
            z_act = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif type == "sigmoid":
            z_act = 1 / (1 + np.exp(-z))
        elif type == "relu":
            z_act = np.maximum(z, 0)
        else:
            z_act = z
        return z_act

    def diff_actFun(self, z, type):
        """
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        """
        if type == "tanh":
            z_diff = 4 / (np.exp(z) + np.exp(-z)) ** 2
        elif type == "sigmoid":
            z_diff = np.exp(-z) / (1 + np.exp(-z)) ** 2
        elif type == "relu":
            z_diff = (np.sign(z) + 1) / 2
        else:
            z_diff = 1
        return z_diff

    def feedforward(self, X, actFun):
        """
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        z3 = np.exp(self.z2)
        self.probs = z3 / np.sum(z3, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        """
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        """
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        data_loss = - np.sum(np.log10(self.probs[range(num_examples), y]))  # / num_examples

        # Add regularization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        """
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        """
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        """
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        """
        num_examples = len(X)
        pL_z2 = self.probs
        pL_z2[range(num_examples), y] -= 1  # pL_z2 is a combination of partial log-likelihood and softmax
        pL_z1 = np.dot(pL_z2, self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type)
        dW2 = np.dot(self.a1.T, pL_z2) / num_examples
        db2 = np.sum(pL_z2, axis=0, keepdims=True) / num_examples
        dW1 = np.dot(X.T, pL_z1) / num_examples
        db1 = np.sum(pL_z1, axis=0, keepdims=True) / num_examples
        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:

        Args:
            epsilon:
        """
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we sigmoiddon't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
        print("Loss after iteration %i: %f" % (num_passes, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        """
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        """
        plot_decision_boundary(lambda x: self.predict(x), X, y)


class Layer(object):
    """
        This class composes the basic elements of the DeepNeuralNetwork
    """

    def __init__(self, nn_input_dim, nn_output_dim):
        """
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        """
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim

        # initialize the weights and biases in the network
        self.W = np.random.randn(self.nn_input_dim, self.nn_output_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_output_dim))

    def feedforward(self, X, actFun):
        """
        feedforward builds a 1-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return: the output of the layer
        """
        self.z = np.dot(X, self.W) + self.b
        return actFun(self.z)

    def backprop(self, X, pL_a, diff_actFun):
        """
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param pL_a: given loss
        :param diff_actFun: the derivative of the activation function
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        """
        num_examples = len(X)
        pL_z = pL_a * diff_actFun(self.z)
        dW = np.dot(X.T, pL_z) / num_examples
        db = np.sum(pL_z, axis=0, keepdims=True) / num_examples
        pL_X = np.dot(pL_z, self.W.T)
        return dW, db, pL_X



class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, nn_input_dim, nn_num_layers, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01,
                 seed=0):
        """
        :param nn_input_dim: input dimension
        :param nn_num_layers: the number of layers
        :param nn_hidden_dim: the number of hidden units
        If it is an integer, then the number of hidden units are the same in each hidden layer
        If it is a list, then each value in the list indicates the number of units in each hidden layer in order
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        """
        self.nn_input_dim = nn_input_dim
        if isinstance(nn_hidden_dim, list):
            self.nn_hidden_dims = nn_hidden_dim
            self.nn_num_layers = len(self.nn_hidden_dims) + 2
        else:
            self.nn_num_layers = nn_num_layers
            self.nn_hidden_dims = [nn_hidden_dim] * (nn_num_layers - 2)
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        # initialize the weights and biases in the network
        np.random.seed(seed)
        if len(self.nn_hidden_dims) == 0:
            layers = [Layer(nn_input_dim, nn_output_dim)]
        else:
            layers = [Layer(nn_input_dim, self.nn_hidden_dims[0])]
            layers.extend([Layer(self.nn_hidden_dims[i], self.nn_hidden_dims[i+1]) for i in range(self.nn_num_layers - 3)])
            layers.append(Layer(self.nn_hidden_dims[-1], nn_output_dim))
        self.layers = layers


    def feedforward(self, X, actFun):
        """
            feedforward builds a 3-layer neural network and computes the two probabilities,
            one for class 0 and one for class 1
            :param X: input data
            :param actFun: activation function
            :return:
        """
        a = X
        self.a = []
        for i in range(len(self.layers) - 1):
            a = self.layers[i].feedforward(a, actFun)
            self.a.append(a)
        a = self.layers[-1].feedforward(a, lambda x: self.actFun(x, type="linear"))
        z = np.exp(a)
        self.probs = z / np.sum(z, axis=1, keepdims=True)
        return None


    def backprop(self, X, y):
        """
            backprop run backpropagation to compute the gradients used to update the parameters in the backward step
            :param X: input data
            :param y: given labels
            :return: dWs, dbs
        """
        num_examples = len(X)
        pL_z = self.probs
        pL_z[range(num_examples), y] -= 1  # pL_z is a combination of partial log-likelihood and softmax
        dWs, dbs = [], []
        a = [X]
        a.extend(self.a)
        for i in range(self.nn_num_layers - 1):
            if i == 0:  # The last layer uses the softmax activation function
                dW, db, pL_z = self.layers[-1-i].backprop(a[-1-i], pL_z,
                                                          lambda x: self.diff_actFun(x, type="linear"))
            else:
                dW, db, pL_z = self.layers[-1-i].backprop(a[-1-i], pL_z,
                                                          lambda x: self.diff_actFun(x, type=self.actFun_type))
            dWs.append(dW)
            dbs.append(db)
        # Revers the lists of dW and db
        dWs.reverse()
        dbs.reverse()
        return dWs, dbs


    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:

        Args:
            epsilon:
        """
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dWs, dbs = self.backprop(X, y)

            for j, layer in enumerate(self.layers):
                # Add derivatives of regularization terms ('b's don't have regularization terms)
                dW = dWs[j] + self.reg_lambda * layer.W
                db = dbs[j]

                # Gradient descent parameter update
                layer.W += -epsilon * dW
                layer.b += -epsilon * db

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            #if print_loss and i % 1000 == 0:
            prediction = self.predict(X)
            accuracy = np.sum(prediction == y) / len(y)

            print("Loss after iteration %i: %f, Accuracy: %f" % (i, self.calculate_loss(X, y), accuracy))
            if accuracy == 1.:
                return 0
        prediction = self.predict(X)
        accuracy = np.sum(prediction == y) / len(y)
        print("Loss after iteration %i: %f, Accuracy: %f" % (num_passes, self.calculate_loss(X, y), accuracy))


    def calculate_loss(self, X, y):
        """
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        """
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        data_loss = - np.sum(np.log10(self.probs[range(num_examples), y]))  # / num_examples

        # Add regularization term to loss (optional)
        for layer in self.layers:
            data_loss += self.reg_lambda / 2 * (np.sum(np.square(layer.W)))
        return (1. / num_examples) * data_loss


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # Save the data into .csv file
    # fp = open("wine.csv", "w")
    # fp.write("Alcohol\tMalic acid\tAsh\tAlcalinity of ash\tMagnesium\tTotal phenols\tFlavanoids\tNonflavanoid phenols\t"
    #          "Proanthocyanins\tColor intensity\tHue\tOD280/OD315 of diluted wines\tProline\tLabels\n")
    # for i in range(len(X)):
    #     X_row = X[i, :]
    #     for item in X_row:
    #         fp.write("%f\t" % item)
    #     fp.write("%d\n" % y[i])
    # fp.close()
    # X2 = PCA(n_components=2).fit_transform(X)
    # scatter = plt.scatter(X2[:, 0], X2[:, 1], s=40, c=y)
    #
    # handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    # plt.legend(handles, labels, loc="upper right")
    # plt.show()

    # Normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    model = DeepNeuralNetwork(nn_input_dim=13, nn_num_layers=5, nn_hidden_dim=3, nn_output_dim=3, actFun_type='sigmoid')
    model.fit_model(X, y)


if __name__ == "__main__":
    main()
