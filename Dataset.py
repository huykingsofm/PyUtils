from torch.utils import data
import torch
import time

class Dataset(data.Dataset):
    def __init__(self, features, labels, tranforms = None):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels
        self.tranforms = tranforms

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = {"features" : self.features[idx], "label" : self.labels[idx]}
        if self.tranforms:
            sample = self.tranforms(sample)
        return sample

    def __call__(self, key):
        if key == "features":
            return self.features
        elif key == "label":
            return self.labels
        raise Exception()

    def get_by_list(self, indices):
        features = torch.Tensor(self("features"))[indices].tolist()
        label = torch.Tensor(self("label"))[indices]
        return Dataset(features, label, self.tranforms)

    def split(self, ratio, seed = None):
        assert (len(ratio) == 2 or len(ratio) == 3) and sum(ratio) <= 1

        if seed == None:
            seed = time.time_ns()
        torch.manual_seed(seed)

        perm = torch.randperm(len(self))

        N = len(self)
        new_ratio = [0]
        new_ratio.extend(ratio)
        for i in range(1, len(new_ratio)):
            new_ratio[i] = new_ratio[i] + new_ratio[i - 1]
        border = [int(x * N) for x in new_ratio]

        datasets = []
        for i in range(1, len(border)):
            start = border[i - 1]
            end   = border[i]
            datasets.append(self.get_by_list(perm[start : end]))

        return tuple(datasets)


class ToTensor():
    def __init__(self, keys = ["features", "label"]):
        self.keys = keys
    def __call__(self, sample):
        tensor_sample = {}
        for key in self.keys:
            if isinstance(sample[key], int):
                sample[key] = [sample[key]]
            tensor_sample[key] = torch.Tensor(sample[key])
        return tensor_sample

if __name__ == "__main__":
    a = [[1, 2], [3, 4], [5, 6], [3, 2], [2, 1], [3, 2], [3, 4]]
    b = [2, 3, 4, 3, 1, 2, 3]
    dataset = Dataset(a, b, tranforms= ToTensor())
    print(dataset)
    loader = data.DataLoader(dataset, batch_size= 1)
    train, valid = dataset.split((0.6, 0.2), seed= 1)
    print(train)