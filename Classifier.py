from .Module import Module
import torch
class Classifier(Module):
    def __metric__(self):
        if self.validating_set == None:
            return None
        
        output = self(self.validating_set[0])
        output = torch.argmax(output, dim= 1)

        n_true = (output.type(self.validating_set[1].type()) == self.validating_set[1]).sum().item()
        return "Accuracy={:.2f}%".format(100 * n_true / self.validating_set[0].shape[0])

if __name__ == "__main__":
    import torch.nn as nn
    X = torch.Tensor([[1, 2], [2, 3], [4, 3]])
    Y = torch.Tensor([2, 2, 2])
    model = Classifier()
    model.add_module("Dense1", nn.Linear(2, 3))
    model.add_module("Softmax", nn.Softmax(dim=1))
    print(model)
    model.validating_set = (X, Y)
    print(model(X))
    print(model.__metric__())