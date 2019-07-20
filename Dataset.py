from torch.utils import data
import torch
import time

class Dataset(data.Dataset):
    def __init__(self, features, labels, tranforms = None):
        assert len(features) == len(labels), "length of size of features and label is mismatched"
        assert isinstance(features, torch.Tensor) and isinstance(labels, torch.Tensor), "data must be a tensor"
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

    def split(self, ratio, seed = None):
        assert (len(ratio) == 2 or len(ratio) == 3) and sum(ratio) <= 1

        if seed is None:
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
            data = self[perm[start : end]]
            datasets.append(Dataset(data["features"], data["label"], self.tranforms))

        return tuple(datasets)

if __name__ == "__main__":
    a = torch.Tensor([[1, 2], [3, 4], [5, 6], [3, 2], [2, 1], [3, 2], [3, 4]])
    b = torch.Tensor([2, 3, 4, 3, 1, 2, 3]).view(-1, 1)
    dataset = Dataset(a, b)
    loader = data.DataLoader(dataset, batch_size= 1)
    x = dataset.split((0.6, 0.2, 0.2), seed= 2)
    print(x)