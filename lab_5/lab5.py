import os
import torch
import torch.nn.functional as F
import torchbearer
import numpy as np

from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchbearer import Trial
from torch import nn


class MyDataset(Dataset):
    def __init__(self, size=5000, dim=40, random_offset=0):
        super(MyDataset, self).__init__()
        self.size = size
        self.dim = dim
        self.random_offset = random_offset

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))

        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)

        while True:
            img = torch.zeros(self.dim, self.dim)
            dx = torch.randint(-10, 10, (1,), dtype=torch.float)
            dy = torch.randint(-10, 10, (1,), dtype=torch.float)
            c = torch.randint(-20, 20, (1,), dtype=torch.float)

            params = torch.cat((dy / dx, c))
            xy = torch.randint(0, img.shape[1], (20, 2), dtype=torch.float)
            xy[:, 1] = xy[:, 0] * params[0] + params[1]

            xy.round_()
            xy = xy[xy[:, 1] > 0]
            xy = xy[xy[:, 1] < self.dim]
            xy = xy[xy[:, 0] < self.dim]

            for i in range(xy.shape[0]):
                x, y = xy[i][0], self.dim - xy[i][1]
                img[int(y), int(x)] = 1
            if img.sum() > 2:
                break

        torch.set_rng_state(rng_state)
        return img.unsqueeze(0), params

    def __len__(self):
        return self.size


class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, (3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(48 * 40 ** 2, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CNNPooling(nn.Module):
    def __init__(self):
        super(CNNPooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(48, 48, (3, 3), stride=1, padding=1)
        self.mp1 = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(48, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.mp1(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CNN3C(nn.Module):
    def __init__(self):
        super(CNN3C, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(48, 48, (3, 3), stride=1, padding=1)
        self.mp1 = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(48, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        idxx = torch.repeat_interleave(
            torch.arange(-20, 20, dtype=torch.float).unsqueeze(0) / 40.0,
            repeats=40,
            dim=0,
        ).to(x.device)
        idxy = idxx.clone().t()
        idx = torch.stack([idxx, idxy]).unsqueeze(0)
        idx = torch.repeat_interleave(idx, repeats=x.shape[0], dim=0)
        x = torch.cat([x, idx], dim=1)
        # for xi in x[0]:
        #     print("")
        #     print(xi)
        # exit()
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.mp1(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    output_path: str = None,
):
    # Define the loss function and the optimiser
    loss_function = nn.MSELoss()
    optimiser = optim.Adam(model.parameters())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function, metrics=["loss", "accuracy"]).to(
        device
    )
    trial.with_generators(train_loader, val_generator=val_loader)
    trial.run(epochs=100)
    if output_path:
        torch.save(model.state_dict(), output_path)
    return trial


def test_model(model: nn.Module, test_loader: DataLoader):
    loss_function = nn.MSELoss()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, None, loss_function, metrics=["loss", "accuracy"]).to(device)
    trial.with_generators(train_loader, test_generator=test_loader)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    return results, trial


def seed(n: int = None):
    if n is None:
        n = np.random.randint(0, 2 ** 32)
    torch.manual_seed(n)
    torch.backends.cudnn.deterministic = True
    print("Seed:", n)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create datasets
    train_data = MyDataset()
    val_data = MyDataset(size=500, random_offset=33333)
    test_data = MyDataset(size=500, random_offset=99999)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3)
    idxx = torch.repeat_interleave(
        torch.arange(-20, 20, dtype=torch.float).unsqueeze(0) / 40.0,
        repeats=40,
        dim=0,
    )
    idxy = idxx.clone().t()
    ax[0].imshow(train_data[0][0][0])
    ax[1].imshow(idxx)
    ax[2].imshow(idxy)
    plt.show()
    # Ex 1
    print("Ex 1:")
    seed(n=7)
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    train = False
    weights_path = "cnn_baseline.weights"
    model = CNNBaseline()
    if train:
        train_model(model, train_loader, val_loader, weights_path)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        model = model.eval()
    test_model(model, test_loader)

    # Build the model
    print("Ex 2:")
    seed(n=7)
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    train = False
    weights_path = "cnn_pooling.weights"
    model = CNNPooling()
    if train:
        train_model(model, train_loader, val_loader, weights_path)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        model = model.eval()
    test_model(model, test_loader)

    # Build the model
    print("Ex 3:")
    seed(n=7)
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    train = False
    weights_path = "cnn_3c.weights"
    model = CNN3C()
    if train:
        train_model(model, train_loader, val_loader, weights_path)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        model = model.eval()
    test_model(model, test_loader)
