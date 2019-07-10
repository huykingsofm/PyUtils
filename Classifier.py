from .Module import Module
import torch
class Classifier(Module):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __metric__(self):
        if self.validating_set == None:
            return None
        
        output = self(self.validating_set[0])
        output = torch.argmax(output, dim= 1)

        n_true = (output == self.validating_set[1].view(output.shape)).sum().item()
        return "Accuracy={:.2f}%".format(100 * n_true/self.validating_set[0].shape[0])