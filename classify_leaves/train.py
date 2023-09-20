import contextlib
import time
import copy
import torch
from torch import nn
import sys
from util.data_loader import BaseDataset, DataLoader, read_csv, albu, ToTensorV2, tqdm, cv
from torch.optim.lr_scheduler import ExponentialLR
from util.net import Net
import torchvision
from sklearn.model_selection import train_test_split


class DummyFile:
    def __init__(self, file):
        if file is None:
            file = sys.stderr
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

@contextlib.contextmanager
def redirect_stdout(file=None):
    if file is None:
        file = sys.stderr
    old_stdout = file
    sys.stdout = DummyFile(file)
    yield
    sys.stdout = old_stdout


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    
    data_csv, unique_labels = read_csv("/home/nrc/classify_leaves/data/train.csv")

    train_data, test_data = train_test_split(data_csv, test_size=0.2, shuffle=False)

    train_transform = albu.Compose([
        albu.Resize(320, 320),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Rotate(limit=180, p=0.7),
        albu.RandomBrightnessContrast(),
        albu.ShiftScaleRotate(
            shift_limit=0.25, scale_limit=0.1, rotate_limit=0
        ),
        albu.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            max_pixel_value=255.0, always_apply=True
        ),
        ToTensorV2(p=1.0),
    ])

    test_transform = albu.Compose([
        albu.Resize(320, 320),
        albu.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            max_pixel_value=255.0, always_apply=True
        ),
        ToTensorV2(p=1.0)
    ])

    train_dataset, test_dataset = BaseDataset(train_data, train_transform), BaseDataset(test_data, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False)

    # Hyperparameters

    epochs = 30
    lr = 0.001
    losses = []
    net = Net()
    net.load_state_dict(torch.load("./checkpoints/best0.pth"))
    best_acc = 0
    # net.apply(weight_init)
    net.to("cuda:0")
    loss = nn.CrossEntropyLoss(reduction="mean")
    optim = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-3)
    scheduler = ExponentialLR(optim, gamma=0.9, verbose=False) # verbose (bool): If True, prints a message to stdout foreach update.

    for epoch in range(epochs):
        acc = 0
        loss_sum = 0
        train_bar = tqdm(train_loader, leave=True)

        net.train()
        for x, y in train_bar:
            x = torch.as_tensor(x, dtype=torch.float)
            x, y = x.to("cuda:0"), y.to("cuda:0")
            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += loss_temp
            optim.zero_grad()
            loss_temp.backward()
            optim.step()
            acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        scheduler.step()
        losses.append(loss_sum.cpu().detach().numpy() / len(train_loader))
        with redirect_stdout():
            print(f"epoch: {epoch}, loss={loss_sum.item()}, 训练集准确度={(acc / ( len(train_loader) * train_loader.batch_size )).item()}", end="")

        test_acc = 0
        net.eval()
        for x, y in tqdm(test_loader):
            x = torch.as_tensor(x, dtype=torch.float)
            x, y = x.to("cuda:0"), y.to("cuda:0")
            y_hat = net(x)
            test_acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        with redirect_stdout():
            print(f"验证集准确度: {(test_acc / (len(test_loader) * test_loader.batch_size)).item()} ")

        if test_acc > best_acc:
            best_model_state = copy.deepcopy(net.state_dict())
            best_acc = test_acc
            torch.save(best_model_state, "./checkpoints/best.pth") # load: the_model.load_state_dict(torch.load(PATH))

    
    
