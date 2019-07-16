from torch.utils import data
import torch

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

    def __call__(self, key= "features"):
        if key == "features":
            return self.features
        elif key == "label":
            return self.labels
        raise Exception()

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
    a = [[1, 2], [3, 4], [5, 6]]
    b = [2, 3, 4]
    dataset = Dataset(a, b, tranforms= ToTensor())
    loader = data.DataLoader(dataset, batch_size= 1)
    print(dataset[2:3])