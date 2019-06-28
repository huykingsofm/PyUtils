version = "1.2"
from random import randint
from math import sqrt
import os
import wx

def get_name_in_path(path:str):
    path = path.rstrip("/").rstrip("\\")
    path = path.replace("\\", "/")
    return path.split("/")[-1]

def GetScreenSize():
    _ = wx.App(False)
    return wx.GetDisplaySize()

def GetCurrentWorkingDir():
    return os.getcwd()

def permutation(N):
    perm = [x for x in range(N)]
   
    for idx in range(N - 1):
        i = randint(idx + 1, N - 1)
        t = perm[i]
        perm[i] = perm[idx]
        perm[idx] = t
        
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
    print(get_name_in_path(".\\huy.txt"))