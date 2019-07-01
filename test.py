import torch
import random
from PyUtils.Utils import array_multiply


if __name__ == "__main__":
    a = torch.randint(1, 4, (10, ))
    print(a)
    dg = BalanceBatchGenerator(a, 1)
    print(dg.generate(100))
    print(dg.generate(3))