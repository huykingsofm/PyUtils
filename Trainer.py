import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
from torch.utils.data import DataLoader
import random
from .pytorch_modelsize.pytorch_modelsize import SizeEstimator
from .Dataset import Dataset
import warnings
warnings.filterwarnings("ignore")

def __get_reasonable_size__(model, size, max_mem, max_time, device, unit= "MB"):
    assert unit == "bit" or unit == "MB"
    model = model.cpu()
    
    size = torch.Tensor([size]).cpu().numpy().reshape(-1)
    size = [int(x) for x in size]
    unit = 0 if unit == "MB" else 1
    
    l = 1
    r = int(1e4)
    while l < r:
        print("\r" + " " * 40 + "\r", end= "")
        print("\rSearch distance is [{}, {}]".format(l, r), end= "")
        m = (l + r) // 2
        unit_size = [m]
        unit_size.extend(size)
        
        start_time = time.time()
        estimator = SizeEstimator(model, unit_size)
        mem = estimator.estimate_size()[unit]
        interval_time = time.time() - start_time
        
        if mem > max_mem or interval_time > max_time:
            r = m - 1
        else:
            l = m + 1

    model.to(device)
    print("")
    assert r > 0, "Memory is shortage"
    return r

class Trainer():
    instruction = """
            validation      : dataset, optional
            criterion       : object or function, obligatory
            optimizer       : nn.optim, obligatory
            metric          : "accuracy" or None, obligatory

            n_epoches       : int, obligatory
            print           : int, optional
            checkpoint      : int, optional
            checkpoint_dir  : str, optional
            history         : "batch" or "epoch", optional

            maximum_memory  : int, optional
            maximum_time    : int, optional
        """
    def __init__(self, model, device, keys= ["features", "label"]):
        self.model = model
        
        self.validation = None
        self.criterion = None
        self.optimizer = None

        self.n_epoches = None
        self.print = 0
        self.checkpoint = 1e10
        self.checkpoint_dir = "./"
        self.history = "epoch"

        assert len(keys) == 2
        self.keys = keys
        self.maximum_memory = 1024  # 1GB
        self.maximum_time = 1       # 1s
        self.device = device
        self.metric = None

    def __metric__(self):
        if self.metric in ["none", None]:
            return None
        
        if self.metric == "accuracy":
            if self.validation is None:
                return None
        
            output = self.model(self.validation[:][self.keys[0]])
            output = torch.argmax(output, dim= 1)

            validation_type = self.validation[:][self.keys[1]].type()
            n_true = (output.type(validation_type) == self.validation[:][self.keys[1]]).sum().item()
            return "Accuracy={:.2f}%".format(100 * n_true / len(self.validation))

    def __print_process__(self, iepoch, ibatch, N):
        print("\rEpoch[{:4d}/{}]\tPercentage= {:2.2f}%"
            .format(iepoch + 1, self.n_epoches, 100 * (ibatch + 1) / N), end ="")

    def __print_all__(self, iepoch, ibatch, loss, epoch_start):
        valid_loss = None
        print("\rEpoch[{:4d}/{}]\tLoss= {:.5f}".format(
            iepoch + 1, 
            self.n_epoches,
            loss.item()
        ), end= "")

        if self.validation is not None:
            output = self.model(self.validation[:][self.keys[0]])
            valid_loss = self.criterion(output, self.validation[:][self.keys[1]])
            print("\t\tValid Loss= {:.6f}".format(
                valid_loss.item()
            ), end= "")
                    
        if self.__metric__() != None:
            print("\t\t{}".format(self.__metric__()), end= "")
        
        print("\t\tElapsed time= {:.2f}s".format(time.time() - epoch_start))
        return valid_loss
    def __get_size_for_test__(self, trainset):
        self.size_for_test = __get_reasonable_size__(
            model= self.model, 
            size= trainset[:][self.keys[0]].shape[1:], 
            max_mem= self.maximum_memory,
            max_time= self.maximum_time,
            device= self.device
        )
        if self.size_for_test > len(trainset):
            self.size_for_test = len(trainset) 

    def __call__(self, trainset, batch_size):
        assert self.criterion and self.optimizer

        self.__get_size_for_test__(trainset)
        print("Safe size for test is {}".format(self.size_for_test))

        dataloader = DataLoader(trainset, batch_size= batch_size, shuffle= True)
        N = len(dataloader)
        
        history_loss = []
        history_valid_loss = []
        process_start = time.time()
        for iepoch in range(self.n_epoches):
            self.model.train()
            epoch_start = time.time()
            for ibatch, batch in enumerate(dataloader):
                self.model.zero_grad()
                self.__print_process__(iepoch, ibatch, N)

                batch_X = batch[self.keys[0]]
                batch_Y = batch[self.keys[1]]
                batch_O = self.model(batch_X)

                loss = self.criterion(batch_O, batch_Y)
                loss.backward()

                self.optimizer.step()

                if self.history == "batch":
                    history_loss.append(loss.item())
                
            self.model.eval()
            start = random.randint(0, len(trainset) - self.size_for_test)
            end = start + self.size_for_test
            X = trainset[start : end][self.keys[0]]
            Y = trainset[start : end][self.keys[1]]
            O = self.model(X)
            loss = self.criterion(O, Y)
            valid_loss = None

            if self.history == "epoch":
                history_loss.append(loss.item())

            if self.print != 0 and (iepoch + 1) % self.print == 0:
                valid_loss = self.__print_all__(iepoch, ibatch, loss, epoch_start)
                if valid_loss:
                    valid_loss = valid_loss.item()
                    history_valid_loss.append(valid_loss)

            if (iepoch + 1) % self.checkpoint == 0:
                self.__save__(loss.item(), valid_loss)
            
        print("\nElapsed time= {:.2f}s".format(time.time() - process_start))
        return history_loss, history_valid_loss
        
    def __save__(self, loss, valid_loss = None):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)    
            print("Create directory {} successfully".format(self.checkpoint_dir))
        
        t = datetime.datetime.now() 
        PATH = self.checkpoint_dir + "/{}-loss={:.4f}".format(t, loss)
        PATH = PATH if valid_loss == None else PATH + "-valid_loss={:.4f}".format(valid_loss)

        torch.save(self.model.state_dict(), PATH)
        print("Model was saved in {}".format(PATH))

    def save(self, PATH):
        torch.save(self.model.state_dict(), PATH)
if __name__ == "__main__":
    a = torch.Tensor([[1, 2], [3, 2], [2, 2], [1, 4], [4, 1], [2, 3]] * 1000)
    b = torch.Tensor([[1], [2], [3], [2], [3], [1]] * 1000)
    dataset = Dataset(a, b)
    model = nn.Sequential()
    model.add_module("1", nn.Linear(2, 1024, bias= False))
    model.add_module("2", nn.ReLU(inplace= True))
    model.add_module("3", nn.Linear(1024, 512, bias= True))
    model.add_module("4", nn.Linear(512, 300, bias= False))
    model.add_module("5", nn.Linear(300, 1))
    
    trainer = Trainer(model, "cpu")
    trainer.validation = dataset
    trainer.criterion = nn.MSELoss(reduction= "mean")
    trainer.optimizer = optim.Adam(model.parameters(), lr= 1e-3)
    trainer.n_epoches = 10
    trainer.print = 1
    trainer.maximum_memory = 0.01


    trainer(dataset, 3)
    #trainer(dataset, 3)
    #print(__get_reasonable_size__(model, 2, 1024, 1, "cpu"))
    