import os
import torch
import random
import timeit

def reset(a):
    a.__reset__()

class NormalLoader():
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

class BalanceDataLoader(NormalLoader):
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
        for i, subset in enumerate(self.sub_labels):
            current_time = int(timeit.timeit() * 1e10)
            random.seed(current_time)
            random.shuffle(self.sub_labels[i])
            

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
    class a():
        def __call__(self, x):
            print(x)
            return x + 1

    t = a()
    print(t(10))