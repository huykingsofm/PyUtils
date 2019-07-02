import time
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from .Utils import array_multiply
import timeit

def start_training(model: nn.Module, optimizer: optim, criterion, loader, training_set:tuple, testing_set:tuple= None,
     batch_size= 64, n_epoches= 10, checkpoint_att:tuple= None, print_att:tuple= None, history_att:tuple= "epoch"):
    """
    Start training a model with some attributes  
    Args:
        model         : A instance of torch.nn.Module
        optimizer     : A instance of torch.optim
        criterion     : A loss function which use for that model
        training_set  : A tuple (training_input, training_output)
        testing_set   : A tuple (testing_input, testing_output)
        batch_size    : A int which is the size of each batches
        n_epoches     : A int which is a number of iteration through over training set
        checkpoint_att: A tuple (save_every: int, DIR)
        print_att     : A tuple ("epoch" or "batch", print_every)
        history_att   : "epoch" or "batch"
    """
    
    all_start = time.time()
    hist_loss = []
    hist_valid_loss = []

    X = training_set[0]
    Y = training_set[1]

    if checkpoint_att != None:
        if len(checkpoint_att) == 1:
            if isinstance(checkpoint_att[0], int):
                checkpoint_att = (checkpoint_att[0], "./")
            elif isinstance(checkpoint_att[0], str):
                checkpoint_att = (1, checkpoint_att[0])                
            else:
                raise Exception("checkpoint_att is in incorrect format")

        try:  
            os.mkdir(checkpoint_att[1])
        except OSError:  
            print ("Creation of the directory %s failed" % checkpoint_att[1])
        else:  
            print ("Successfully created the directory %s " % checkpoint_att[1])

    if print_att == None:
        print_att = ("none", 0)
    elif len(print_att) == 1:
        if isinstance(print_att[0], str):
            raise Exception("print_att is need a print_every more than where it print")
        else:
            print_att = ("epoch", print_att[0])

    print("Training size is {}".format(training_set[0].shape[0]))
    if testing_set != None:
        print("Testing size is {}".format(testing_set[0].shape[0]))
    print("Batch size is {}".format(batch_size))
    print("The number of epoches is {}".format(n_epoches))
    if checkpoint_att != None:
        print("Checkpoint will be saved at {} after every {} epoch(s)".format(checkpoint_att[1], checkpoint_att[0]))
    else:
        print("checkpoint is not set")
    if print_att[0] != "none":
        print("print after every {} {}".format(print_att[1], print_att[0]))
    else:
        print("print is not set")
    print("history loss is calculated after every {}\n".format(history_att))
    print("\nThe process is about to run....")
    time.sleep(1)
 
    for iepoch in range(n_epoches):
        if loader == None:
            loader = Loader(training_set[0].shape[0])

        model.train(True)
        start = time.time()
        for istart in range(0, X.shape[0], batch_size):
            if print_att[0] is not "batch":
                print("\rEpoch[{:4d}/{}]\tPercentage= {:2.2f}%"
                    .format(iepoch + 1, n_epoches, (istart + batch_size) * 100 / X.shape[0]), end ="")
            ids = loader.get_batch_idx(batch_size)

            batch_X = X[ids]
            batch_Y = Y[ids]

            batch_O = model(batch_X)

            loss = criterion(batch_O, batch_Y.view(batch_O.shape))
            loss.backward()

            optimizer.step()

            if history_att == "batch":
                hist_loss.append(loss.item())

            if print_att[0] == "batch" and (istart + 1) % print_att[1] == 0:
                print("Epoch[{:4d}/{}]\tBatch[{:4d}/{}]\tLoss= {:8.6f}".format(
                    iepoch + 1, n_epoches,
                    istart + 1, X.shape[0],
                    loss.item(),
                ))

        model.eval()
        output = model(X)
        loss = criterion(output, Y.view(output.shape))

        if history_att == "epoch":
            hist_loss.append(loss.item())
        
        if (print_att[0] == "batch" and (iepoch + 1) % print_att[1] == 0):
            print("\rEpoch[{:4d}/{}]\tLoss= {:.6f}".format(
                iepoch + 1, n_epoches,
                loss.item()
            ), end= "")
            
            if testing_set != None:
                output = model(testing_set[0])
                valid_loss = criterion(output, testing_set[1].view(output.shape))
                print("\t\tValid Loss= {:.6f}".format(
                    valid_loss.item()
                ), end= "")
                hist_valid_loss.append(valid_loss.item())
            
            print("\t\tElapsed time= {:.2f}s".format(time.time() - start))
        
        if (iepoch + 1) % checkpoint_att[0] == 0:
            t = datetime.datetime.now()
            PATH = checkpoint_att[1] + "/{}-loss={:.4f}".format(t, loss.item())
            torch.save(model.state_dict(), PATH)
            print("Model was saved in {}".format(PATH))

    all_end = time.time()
    print("")
    print("Elapsed time= {:.2f}s\tAvarage elapsed time per epoch= {:2f}s"
        .format(all_end - all_start, (time.time() - all_start) / n_epoches))
    return model, hist_loss, hist_valid_loss

class Loader():
    def __init__(self, data_size):
        self.data_size = data_size
        self.__reset__()

    def __reset__(self):
        self.shuffle_idx = list(range(self.data_size))
        random.shuffle(self.shuffle_idx)
        self.cursor = 0

    def get_batch_idx(self, size):
        ret = torch.LongTensor(self.shuffle_idx[self.cursor: self.cursor + size])
        self.cursor += size
        return ret

class BalanceDataLoader(Loader):
    def __init__(self, labels, d, device):
        self.device = device
        # thống kê
        min_value = int(labels.min())
        max_value = int(labels.max())

        values = list(range(min_value - 1, max_value + 1, d))
        self.size = len(labels)
        
        # chia nhỏ labels thành nhiều sub-labels nhỏ, mỗi sub-labels đại diện cho một đoạn giá trị
        self.sub_labels = []
        backup = labels.clone()
        self.N = 0
        for i, value in enumerate(values[1:]):
            position = list(torch.nonzero(backup <= value).view(-1).cpu().numpy())
            if len(position) > 0:
                self.sub_labels.append(position)
            backup[position] += 1e10
        self.N = len(self.sub_labels)
        self.__reset__()

    def __reset__(self):
        self.cursor = [0] * len(self.sub_labels)
        for subset in self.sub_labels:
            current_time = int(timeit.timeit() * 1e10)
            random.seed(current_time)
            random.shuffle(subset)
            

    def __get_position_of_subset__(self, isubset, l):
        ret = []

        while l > 0:
            if l > len(self.sub_labels[isubset]) - self.cursor[isubset]:
                d = len(self.sub_labels[isubset]) - self.cursor[isubset]
                l = l - d
            else:
                d = l
                l = 0
            start = self.cursor[isubset]
            end = start + d
            ret.extend(self.sub_labels[isubset][start : end])
            self.cursor[isubset] = (self.cursor[isubset] + d) % len(self.sub_labels[isubset])

            if self.cursor[isubset] == 0:
                random.shuffle(self.sub_labels[isubset])

        return ret

    def get_batch_idx(self, size):
        avg = size // self.N
        last = size - avg * (self.N - 1)

        who_last = random.randint(0, self.N - 1)
        position = []
        for i in range(self.N):
            if len(self.sub_labels[i]) == 0:
                continue
            if i == who_last:
                t = self.__get_position_of_subset__(i, last)
            else:
                t = self.__get_position_of_subset__(i, avg)
            position.extend(t)
        return torch.LongTensor(position).to(self.device)
if __name__ == "__main__":
    i =              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = torch.Tensor([1, 2, 3, 2, 1, 3, 2, 3, 5, 5])
    dg = BalanceDataLoader(a, 1, torch.device("cpu"))
    print(dg.get_batch_idx(11))