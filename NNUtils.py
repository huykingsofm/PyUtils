import time
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim

def start_training(model: nn.Module, optimizer: optim, criterion, training_set:tuple, testing_set:tuple= None,
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

    print("Set up completely")
    print("training size= {}\ttesting size= {}".format(training_set[0].shape[0], testing_set[0].shape[0]))
    print("batch size= {}".format(batch_size))
    print("n_epoches= {}".format(n_epoches))
    if checkpoint_att != None:
        print("checkpoint save at {} after every {} epoch(s)".format(checkpoint_att[1], checkpoint_att[0]))
    else:
        print("checkpoint is not set")
    if print_att[0] != "none":
        print("print after every {} {}".format(print_att[1], print_att[0]))
    else:
        print("print is not set")
    print("history loss is calculated after every {}\n".format(history_att))
    time.sleep(1)
 
    for iepoch in range(n_epoches):
        shuffer_idx = torch.randperm(X.shape[0])
        
        model.train(True)
        start = time.time()
        for istart in range(0, X.shape[0], batch_size):
            if print_att[0] is not "batch":
                print("\rEpoch[{:4d}/{}]\tPercentage= {:2.2f}%"
                    .format(iepoch + 1, n_epoches, istart * 100 / X.shape[0]), end ="")

            batch_X = X[shuffer_idx[istart: istart + batch_size]]
            batch_Y = Y[shuffer_idx[istart: istart + batch_size]]

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
        if print_att[0] is not "batch":
            print("")

        model.eval()
        output = model(X)
        loss = criterion(output, Y.view(output.shape))

        if history_att == "epoch":
            hist_loss.append(loss.item())
        
        if (print_att[0] == "batch" and (iepoch + 1) % print_att[1] == 0) or testing_set != None:
            print("Epoch[{:4d}/{}]\tLoss= {:.6f}".format(
                iepoch + 1, n_epoches,
                loss.item()
            ), end= "")
            
            if testing_set != None:
                output = model(testing_set[0])
                loss = criterion(output, testing_set[1].view(output.shape))
                print("\tValid Loss= {:.6f}".format(
                    loss.item()
                ), end= "")
            
            print("\tElapsed time= {:.2f}s".format(time.time() - start))
        
        if (iepoch + 1) % checkpoint_att[0] == 0:
            t = datetime.datetime.now()
            PATH = checkpoint_att[1] + "/{}-loss={:.4f}".format(t, loss.item())
            torch.save(model.state_dict(), PATH)
            print("Module was saved in {}".format(PATH))
    all_end = time.time()
    print("Elapsed time= {:.2f}s\tAvarage elapsed time per epoch= {:2f}s"
        .format(all_end - all_start, (time.time() - all_start) / n_epoches))
    return model, hist_loss