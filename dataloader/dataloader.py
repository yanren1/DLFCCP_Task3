import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SampleDataset(Dataset):
    def __init__(self, root_dir, file_name):
        super(SampleDataset, self).__init__()

        self.root_dir = root_dir
        self.file_name = file_name
        self.samples = self.__read_xlsx()

    def __getitem__(self, index):
        samples = self.samples[index]
        # sample, target = samples[:-3],samples[-3:]

        return samples[1:], samples[0].long()

    def __len__(self):
        return len(self.samples)

    def __read_xlsx(self):
        f_pth = os.path.join(self.root_dir, self.file_name)
        # f_pth = os.path.join(root_dir, 'data.xlsx')
        df = pd.read_csv(f_pth,usecols=['body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg', 'price','make'])
        # ['body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg', 'price']
        # samples = torch.from_numpy(df.to_numpy()).float()
        # [make, body - style, wheel - base, engine - size, horsepower, peak - rpm, highway - mpg]
        # print(samples)
        # samples = df.to_dict(orient='list')


        samples = torch.from_numpy(df.to_numpy()).float()
        return samples


class XORDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.Y = [0, 1, 1, 0]
        self.data = self.generate_data()

    def generate_data(self):
        inputs = []
        labels = []
        for i in range(self.size):
            idx = random.randint(0,3)
            inputs.append(self.X[idx])
            labels.append(self.Y[idx])

        return {'inputs': inputs, 'labels': labels}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # sample = {'input': self.data['inputs'][idx], 'label': self.data['labels'][idx]}
        return torch.tensor(self.data['inputs'][idx]).float(), torch.tensor(self.data['labels'][idx]).float()

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader, random_split
#     train_ratio = 0.85
#     dataset = SampleDataset(root_dir='../data', file_name='output_file.csv')
#     train_size = int(train_ratio * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=int(len(train_dataset)),
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=True,
#     )
#     val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

