__author__ = 'tan_nguyen'

import numpy as np
import os
import argparse
import pickle
import shutil
import csv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.utils.data
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from collections import OrderedDict


class DatasetLoader(torch.utils.data.Dataset):

    def __init__(self, name_list):
        cell_data = []
        for name in name_list:
            with open(name) as csvfile:
                csv_reader = csv.reader(csvfile)
                cell_header = next(csv_reader)
                for row in csv_reader:
                    cell_data.append(row)

        cell_data = [[float(x) for x in row] for row in cell_data]
        cell_data = np.array(cell_data, dtype=np.float32)
        #cell_data = (cell_data - np.mean(cell_data, axis=0)) / np.std(cell_data, axis=0)
        self.cell_label = torch.LongTensor(cell_data[:, -1])
        self.cell_data = torch.FloatTensor(cell_data[:, 1:-1])
        self.length = len(self.cell_data)

    def __getitem__(self, item):
        return self.cell_data[item, :], self.cell_label[item]

    def get_all(self):
        return self.cell_data, self.cell_label

    def __len__(self):
        return self.length


def construct_linear_block(nn_input_dim, nn_output_dim, actFun_type):
    sequential = [("batch_norm", nn.BatchNorm1d(nn_input_dim)), ("linear_block", nn.Linear(nn_input_dim, nn_output_dim))]
    if actFun_type == "tanh":
        sequential.append(("tanh", nn.Tanh()))
    elif actFun_type == "sigmoid":
        sequential.append(("sigmoid", nn.Sigmoid()))
    elif actFun_type == "relu":
        sequential.append(("relu", nn.ReLU()))
    elif actFun_type == "softmax":
        sequential.append(("softmax", nn.Softmax()))

    return nn.Sequential(OrderedDict(sequential))


class DeepNeuralNetwork(nn.Module):
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
        super(DeepNeuralNetwork, self).__init__()
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
            layers = [("layer_0", construct_linear_block(nn_input_dim, nn_output_dim, "softmax"))]
        else:
            layers = [("layer_0", construct_linear_block(nn_input_dim, self.nn_hidden_dims[0], actFun_type))]
            layers.extend(
                [("layer_%d" % (i+1),
                  construct_linear_block(self.nn_hidden_dims[i], self.nn_hidden_dims[i + 1], actFun_type))
                 for i in range(self.nn_num_layers - 3)])
            layers.append(("layer_%d" % (self.nn_num_layers - 2),
                           construct_linear_block(self.nn_hidden_dims[-1], nn_output_dim, "softmax")))

        self.add_module("layers", nn.Sequential(OrderedDict(layers)))

    def forward(self, X):
        """
            feedforward builds a n-layer neural network and computes the two probabilities,
            one for class 0 and one for class 1
            :param X: input data
            :return:
        """
        layers = self.__getattr__("layers")
        y = layers(X)
        return y

    def predict(self, X):
        probs = self.forward(X)
        return torch.argmax(probs, 1)


def load_nets(dir, nets):
    '''
        Load pretrained weights saved with save_nets()

        :param dir: folder where nets were saved
        :param nets: dictionary of pytorch modules to load
        :returns: nets
    '''
    for key, val in nets.items():
        state_dict = torch.load(dir + '/' + key + '.net')
        val.load_state_dict(state_dict)

    return nets


def save_nets(dir, nets, to_pickle=None, overwrite=True):
    '''
        Create folder and save nets and pickle

        :param dir: folder to save dataset
        :param nets: dictionary of pytorch modules to save
        :param to_pickle: any pickable object, us to save other checkpoint data
        :param overwrite: overwrite dir if exists
    '''
    if (overwrite):
        try:
            shutil.rmtree(dir)
        except FileNotFoundError:
            pass
    os.makedirs(dir)

    for key, val in nets.items():
        torch.save(val.state_dict(), dir + '/' + key + '.net')

    if to_pickle is not None:
        pickle.dump(to_pickle, open(dir + '/TrainingLog.pkl', 'wb'))


def main(args):
    # generate and visualize Make-Moons dataset
    dataset = DatasetLoader(["Tstable.csv", "Tunstable.csv"])
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, pin_memory=True, num_workers=5)


    model = DeepNeuralNetwork(nn_input_dim=args.input_dim, nn_num_layers=args.num_layers, nn_hidden_dim=args.hidden_dim,
                              nn_output_dim=args.output_dim, actFun_type='tanh')
    model.cuda()
    loss = nn.NLLLoss()

    load_dir = os.path.join(args.output_dir, "ckpts")
    start_idx = 0
    if os.path.exists(os.path.join(args.output_dir, "ckpts")):
        ckpts = os.listdir(load_dir)
        if ckpts:
            ckpts.sort()
            start_idx = int(ckpts[-1].split("-")[-1].split(".")[0])
            load_nets(load_dir, {
                "DNN-%d" % start_idx: model
            })

    log = SummaryWriter(os.path.join(args.output_dir, "log"))
    log.add_graph(model, input_to_model=torch.rand(10, args.input_dim).cuda())

    optimizer = optim.Adam(model.parameters())
    features, target = dataset.get_all()
    features = features.cuda()
    target = target.cuda()
    for i in range(start_idx, args.epochs):
        for batch_idx in range(len(dataloader)):

            optimizer.zero_grad()
            #features, target = next(dataloader.__iter__())

            #features = Variable(features, requires_grad=True)
            prediction = model(features)

            if i % 1000 == 0:
                accuracy = torch.sum(model.predict(features) == target) / len(target)
                print("Epoch %d, accuracy=%f" % (i, accuracy))

                save_nets(os.path.join(args.output_dir, "ckpts"), {
                    "DNN-%d" % i: model
                })

            loglikelihood = loss(prediction, target)
            loglikelihood.backward()
            optimizer.step()

    save_nets(os.path.join(args.output_dir, "ckpts"), {
        "DNN-%d" % args.epochs: model
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=39, help='input dimension')
    parser.add_argument("--num_layers", type=int, default=4, help='the number of layers')
    parser.add_argument("--hidden_dim", type=int, default=100, help='the number of hidden units')
    parser.add_argument("--output_dim", type=int, default=2, help='the number of output units')
    parser.add_argument("--output_dir", type=str, default="cells", help='the output directory')
    parser.add_argument("--epochs", type=int, default=20000, help='the number of training epochs')
    args = parser.parse_args()
    main(args)
