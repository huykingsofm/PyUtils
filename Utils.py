version = "1.2"
from random import randint
import random
from math import sqrt
import os
import timeit

def array_multiply(arr: list, multiplier: float, shuffle= True, shuffle_last= True):
    integer = int(multiplier)
    resident = multiplier - integer
    
    last = arr.copy()
    if shuffle_last:
        random.shuffle(last)
    
    new_arr = []
    for i in range(integer):
        t = arr.copy()
        if shuffle:
            random.seed(int(timeit.timeit() * 1e10))
            random.shuffle(t)
        new_arr.extend(t)
    
    last_len = int(resident * len(arr))
    new_arr.extend(last[:last_len])
    return new_arr

def get_name_in_path(path:str):
    path = path.rstrip("/").rstrip("\\")
    path = path.replace("\\", "/")
    return path.split("/")[-1]

def GetScreenSize():
    import wx
    _ = wx.App(False)
    return wx.GetDisplaySize()

def GetCurrentWorkingDir():
    return os.getcwd()

def permutation(N):
    perm = [range(N)]
    random.shuffle(perm)
    return perm

def squarize(inp):
    if  sqrt(len(inp)) != int(sqrt(len(inp))):
        raise Exception("input len ({}) can not conver to square matrix".format(len(inp)))
    
    square = []
    N = int(sqrt(len(inp)))
    for i in range(N):
        square.append([])
        for j in range(N):
            square[i].append(inp[i * N + j])
            
    return square
     

def flatten(inp):
    return __flatten__(inp, [])
    
def __flatten__(inp, out : list):
    if type(inp) != type([]):
        out.append(inp)
        return out
    
    for element in inp:
        __flatten__(element, out)
    return out

if __name__ == "__main__":
    print(array_multiply([1, 2, 3], 10.4, shuffle= False, shuffle_last= True))