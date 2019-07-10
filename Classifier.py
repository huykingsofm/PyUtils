from .Module import Module
import torch
class Classifier(Module):
    def __metric__(self):
        if self.validating_set[0] is None:
            return None
        
        output = self(self.validating_set[0])
        output = torch.argmax(output, dim= 1)

        n_true = (output.type(self.validating_set[1].type()) == self.validating_set[1]).sum().item()
        return "Accuracy={:.2f}%".format(100 * n_true / self.validating_set[0].shape[0])

if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    X = torch.Tensor([[1, 2], [2, 3], [4, 3], [3, 2], [1, 4], [2, 4], [3, 4], [2, 1], [4, 2]])
    Y = torch.Tensor([1, 0, 3, 2, 3, 1, 2, 2, 0]).type(torch.LongTensor)
    model = Classifier()
    model.add_module("1", nn.Linear(2, 100, bias= False))
    model.add_module("2", nn.ReLU(inplace= True))
    model.add_module("3", nn.Linear(100, 20, bias= True))
    model.add_module("4", nn.ReLU(inplace= True))
    model.add_module("5", nn.Linear(20, 4, bias= True))
    model.add_module("6", nn.Softmax(dim= 1))
    
    model.optimizer = optim.Adam(model.parameters(), lr= 1e-4, weight_decay= 1e-5)
    model.criterion = nn.CrossEntropyLoss(reduction= "mean")
    model.validating_set = (X, Y)
    
    model.print = 10
    model.batch_size = 2
    model.n_epoches = 100000
    
    #print(model.state_dict().values())
    model.fit(X, Y)