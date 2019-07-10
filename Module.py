import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
from .BatchLoader import NormalLoader, reset

class Module(nn.Module):
    instruction = """
            validating_set  : tuple (data, label) optional
            criterion       : object or function, obligatory
            optimizer       : nn.optim, obligatory
            batchloader     : batchloader object, optional

            batch_size      : int, obligatory
            n_epoches       : int, oblidgatory
            print           : int, optional
            checkpoint      : int, optional
            checkpoint_dir  : str, optional
            history         : str in ("batch", "epoch"), optional
        """
        
    def __init__(self):
        super(Module, self).__init__()
        
        self.validating_set = (None, None)
        self.criterion = None
        self.optimizer = None
        self.batchloader = None

        self.batch_size = 64
        self.n_epoches = 10
        self.print = 0
        self.checkpoint = 1e10
        self.checkpoint_dir = "./"
        self.history = "epoch"


        self.main = nn.Sequential()
    def forward(self, X):
        return self.main(X)

    def add_module(self, name, module):
        self.main.add_module(name, module)
    
    def __metric__(self):
        return None

    def __check__(self):
        if self.criterion == None:
            raise Exception("Criterion is miss")

        if self.optimizer == None:
            raise Exception("Optimizer is miss")

    def fit(self, X, Y):
        self.__check__()
        if self.batchloader == None:
            self.batchloader = NormalLoader(data_size= X.shape[0])

        history_loss = []
        history_valid_loss = []
        process_start = time.time()
        for iepoch in range(self.n_epoches):
            reset(self.batchloader)
            self.train()
            epoch_start = time.time()
            for ibatch in range(0, X.shape[0], self.batch_size):
                self.zero_grad()
                print("\rEpoch[{:4d}/{}]\tPercentage= {:2.2f}%"
                    .format(iepoch + 1, self.n_epoches, (ibatch + self.batch_size) * 100 / X.shape[0]), end ="")

                shuffle_id = self.batchloader.get_batch_idx(self.batch_size)
                batch_X = X[shuffle_id]
                batch_Y = Y[shuffle_id]
                batch_O = self(batch_X)

                loss = self.criterion(batch_O, batch_Y)
                loss.backward()

                self.optimizer.step()

                if self.history == "batch":
                    history_loss.append(loss.item())
            
            self.eval()
            O = self(X)
            loss = self.criterion(O, Y)
            valid_loss = None
            if self.history == "epoch":
                history_loss.append(loss.item())

            if self.print != 0 and (iepoch + 1) % self.print == 0:
                print("\rEpoch[{:4d}/{}]\tLoss= {:.6f}".format(
                    iepoch + 1, self.n_epoches,
                    loss.item()
                ), end= "")

                if self.validating_set[0] is not None:
                    output = self(self.validating_set[0])
                    valid_loss = self.criterion(output, self.validating_set[1])
                    print("\t\tValid Loss= {:.6f}".format(
                        valid_loss.item()
                    ), end= "")
                    history_valid_loss.append(valid_loss.item())
                
                if self.__metric__() != None:
                    print("\t\t{}".format(self.__metric__()), end= "")

                print("\t\tElapsed time= {:.2f}s".format(time.time() - epoch_start))
            if (iepoch + 1) % self.checkpoint == 0:
                self.__save__(loss, valid_loss)

        print("\nElapsed time= {:.2f}s".format(time.time() - process_start))
        return history_loss, history_valid_loss
        
    def __save__(self, loss, valid_loss = None):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)    
            print("Create directory {} successfully".format(self.checkpoint_dir))

        criterion = self.criterion
        output = self(self.training_set[0])
        loss = criterion(output, self.training_set[0].view(output.shape)).item()
        
        t = datetime.datetime.now() 
        PATH = self.checkpoint_dir + "/{}-loss={:.4f}".format(t, loss)
        PATH = PATH if valid_loss == None else PATH + "-valid_loss={:.4f}".format(valid_loss)

        torch.save(self.state_dict(), PATH)
        print("Model was saved in {}".format(PATH))

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)
if __name__ == "__main__":
    model = Module()
    print(Module.instruction)