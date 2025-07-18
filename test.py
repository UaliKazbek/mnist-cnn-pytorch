import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt
import numpy as np

import json
from tqdm import tqdm
from PIL import Image


plt.style.use('dark_background')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MNISTDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = dir_list
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = os.path.basename(path_dir)

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))

            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = Image.open(file_path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, ), std=(0.5, ))
    ]
)

test_data = MNISTDataset(r"C:\Users\STARLINECOMP\PycharmProjects\Pytorch\content\mnist\testing", transform=transform)

test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class MyModel(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.l1 = nn.Linear(inp, 128, bias=False)
        self.act = nn.ReLU()
        self.bathnorm = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, out)


    def forward(self, inp):
        x = self.l1(inp)
        x = self.act(x)
        x = self.bathnorm(x)
        out = self.l2(x)
        return out


param_model = torch.load(r'C:\Users\STARLINECOMP\PycharmProjects\Pytorch\model_state_dict_29_posle_bias=False.pt')

model = MyModel(784, 10).to(device)
loss_model = nn.CrossEntropyLoss()

model.load_state_dict(param_model['model_state_dict'])

test_loss = []
test_acc = []

model.eval()
with torch.no_grad():
    running_test_loop = []
    true_answer = 0
    for x, targets in test_loader:
        x = x.reshape(-1, 28 * 28).to(device)
        targets = targets.to(torch.long).to(device)

        pred = model(x)
        loss = loss_model(pred, targets)

        running_test_loop.append(loss.item())
        mean_test_loss = sum(running_test_loop) / len(running_test_loop)

        true_answer += (pred.argmax(dim=1) == targets).sum().item()

    running_test_acc = true_answer / len(test_data)

    test_loss.append(mean_test_loss)
    test_acc.append(running_test_acc)


print(f'test_loss={mean_test_loss:.4f}, train_acc={running_test_acc:.4f}')


