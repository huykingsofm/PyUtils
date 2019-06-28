import torch
import torch.nn as nn
import torch.optim as optim

class FCNN(nn.Module):
    VALID_LAYERS = ["sigmoid", "tanh", "relu", "softmax", "batchnorm", "dropout"]
    def __init__(self, network: list):
        super(FCNN, self).__init__()
        for i in range(len(network)):
            if isinstance(network[i], int) is False and isinstance(network[i], str) is True:
                network[i] = (network[i], )

        self.network = network
        self.main = nn.Sequential()
        last_layer_size = 0

        for i, layer in enumerate(network):
            if isinstance(layer, int):
                layer_size = layer
                if last_layer_size is not 0:
                    self.main.add_module(str(i), nn.Linear(last_layer_size, layer_size, bias= True))
                last_layer_size = layer_size
            
            if isinstance(layer, tuple):
                kind_of_layer = layer[0]
                if kind_of_layer not in FCNN.VALID_LAYERS:
                    raise Exception("Kind of layer({}) is not invalid".format(kind_of_layer))

                if kind_of_layer == "sigmoid":
                    module = nn.Sigmoid()
                elif kind_of_layer == "tanh":
                    module = nn.Tanh()
                elif kind_of_layer == "softmax":
                    module = nn.Softmax(dim= -1)
                elif kind_of_layer == "relu":
                    module = nn.ReLU(inplace= True)
                elif kind_of_layer == "batchnorm":
                    module = nn.BatchNorm1d(last_layer_size)
                elif kind_of_layer == "dropout":
                    p = layer[1] if len(layer) == 2 else 0.5
                    module = nn.Dropout(p= p, inplace= True)
                self.main.add_module(str(i), module)
    def forward(self, X):
        return self.main(X)

    def super_simple_graph(self):
        graph = []
        for i in range(len(self.network)):
            if isinstance(self.network[i], int):
                layer = str(self.network[i])
            else:
                layer = self.network[i][0]
            graph.append(layer)
            
        return "->".join(graph)
