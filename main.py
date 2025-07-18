import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt

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
                self.classes = sorted(dir_list)
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

train_data = MNISTDataset(r"C:\Users\STARLINECOMP\PycharmProjects\Pytorch\content\mnist\training", transform=transform)
test_data = MNISTDataset(r"C:\Users\STARLINECOMP\PycharmProjects\Pytorch\content\mnist\testing", transform=transform)

train_data, val_data = random_split(train_data, [0.7, 0.3])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class MyModel(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, 32, 3),  # inp(1, 28, 28) -> out(32, 26, 26)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),  # inp(32, 26, 26) -> out(64, 24, 24)
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

        self.liner = nn.Sequential(
            nn.Linear(64 * 24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, out)
        )


    def forward(self, inp):
        x = self.conv(inp)
        x = self.flatten(x)
        out = self.liner(x)
        return out


model = MyModel(1, 10).to(device)
loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)


EPOCHS = 50
train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
best_loss = None
count = 0

for epoch in range(EPOCHS):

    model.train()
    running_train_loop = []
    true_answer = 0
    train_loop = tqdm(train_loader, leave=False)
    for x, targets in train_loop:
        x = x.to(device)
        targets = targets.to(torch.long).to(device)

        pred = model(x)
        loss = loss_model(pred, targets)

        opt.zero_grad()
        loss.backward()

        opt.step()

        running_train_loop.append(loss.item())
        mean_train_loss = sum(running_train_loop)/len(running_train_loop)

        true_answer += (pred.argmax(dim=1) == targets).sum().item()

        train_loop.set_description(f'EPOCHS [{epoch+1}/{EPOCHS}], train_loss{mean_train_loss:.4f}')

    running_train_acc = true_answer / len(train_data)
    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)

    model.eval()
    with torch.no_grad():
        running_val_loop = []
        true_answer = 0
        for x, targets in val_loader:
            x = x.to(device)
            targets = targets.to(torch.long).to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            running_val_loop.append(loss.item())
            mean_val_loss = sum(running_val_loop) / len(running_val_loop)

            true_answer += (pred.argmax(dim=1) == targets).sum().item()

        running_val_acc = true_answer / len(val_data)
        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

    lr_scheduler.step(mean_val_loss)
    lr = lr_scheduler._last_lr[0]
    lr_list.append(lr)


    print(f'Epoch [{epoch+1}/{EPOCHS}], train_loss={mean_train_loss:.4f}, train_acc={running_train_acc:.4f}, val_loss={mean_val_loss:.4f}, val_acc={running_val_acc:.4f}, lr={lr:.4f}')

    if best_loss is None:
        best_loss = mean_val_loss


    if mean_val_loss < best_loss:
        best_loss = mean_val_loss
        count = 0

        checkpoint = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'EPOCHS': EPOCHS,
            'save_epoch': epoch
        }

        torch.save(checkpoint, f'model_state_dict_{epoch+1}.pt')
        print(f'на {epoch+1} эпохе модель сохранила значение функция потерь на валидации {mean_val_loss:.4f}')
    else:
        count +=1

    if count > 10:
        print(f'на {epoch} эпохе обучение остановилось значение валидации {mean_val_loss:.4f}')
        break


plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['loss_train', 'loss_val'])
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['train_acc', 'val_acc'])
plt.show()
