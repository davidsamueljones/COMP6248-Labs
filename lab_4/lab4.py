import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle

from torch import Tensor
from torch.utils.data import DataLoader
from torchbearer import Trial
from torchbearer.callbacks import PyCM
from torchvision.datasets import MNIST
from pandas import DataFrame
from torchbearer.callbacks import LiveLossPlot
from typing import List, Dict


def get_figure_dir() -> str:
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    figure_dir = os.path.join(cur_dir, "report", "figures")
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def save(fig: plt.Figure, name: str):
    figure_dir = get_figure_dir()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, name + ".pgf"))
    fig.savefig(os.path.join(figure_dir, name + ".pdf"))


def to_pandas_seaborn(
    normalize=False, title="Confusion matrix", annot=False, cmap="YlGnBu"
):
    def handler(cm, state):
        plt.figure()
        string_state = {
            str(key): state[key] for key in state.keys()
        }  # For string formatting
        if normalize == True:
            df = DataFrame(cm.normalized_matrix).T.fillna(0)
        else:
            df = DataFrame(cm.matrix).T.fillna(0)
        ax = sns.heatmap(df, annot=annot, cmap=cmap)
        ax.set_title(title.format(**string_state))
        ax.set(xlabel="Predict", ylabel="Actual")

    return handler


class SingleHiddenLayerMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(SingleHiddenLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = F.softmax(out, dim=1)
        return out


def history_list(key: str, history: Dict) -> List[float]:
    return list(map(lambda x: x[key], history))
    
def history_plots(history: Dict, file_prefix: str):
    fig, ax = plt.subplots(1, 1)
    epoch_range = list(range(len(history)))
    val_epoch_range = list(range(1, len(history)+1))

    def history_list(key: str) -> List[float]:
        return list(map(lambda x: x[key], history))

    ax.plot(epoch_range, history_list("running_loss"))
    ax.plot(epoch_range, history_list("val_loss"))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    fig.set_size_inches(w=3.3, h=1.5)
    save(fig, "_".join([file_prefix, "loss"]))

    fig, ax = plt.subplots(1, 1)
    ax.plot(epoch_range, history_list("running_acc"), label="Training")
    ax.plot(epoch_range, history_list("val_acc"), label="Validation")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("epoch")
    ax.set_ylabel("acc")
    ax.legend(loc=4, prop={"size": 8})
    fig.set_size_inches(w=3.3, h=1.5)
    save(fig, "_".join([file_prefix, "acc"]))


def run(hidden_size: int, file_prefix: str, epochs: int = 20):
    # Flatten 28*28 images to a 784 vector for each image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert to tensor
            transforms.Lambda(lambda x: x.view(-1)),  # flatten into vector
        ]
    )
    trainset = MNIST(".", train=True, download=True, transform=transform)
    testset = MNIST(".", train=False, download=True, transform=transform)
    data_size = torch.numel(trainset[0][0])
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=False)
    testloader = DataLoader(testset, batch_size=128, shuffle=True, drop_last=False)

    model = SingleHiddenLayerMLP(data_size, hidden_size, 10)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cm = (
        PyCM()
        .on_train()
        .with_handler(
            to_pandas_seaborn(normalize=True, title="Confusion Matrix: {epoch}")
        )
    )
    callbacks = [cm]
    trial = Trial(
        model,
        optimiser,
        loss_function,
        metrics=["loss", "accuracy"],
        callbacks=callbacks,
    )
    trial.to(device)
    trial.with_generators(trainloader, val_generator=testloader, val_steps=1)
    history = trial.run(epochs=epochs)

    return history


def load_history(path: str):
    history = torch.load(path)
    prefix = path.split("_")[0]
    history_plots(history, prefix)


if __name__ == "__main__":
    hidden_sizes = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 10000, 100000, 250000]
    load = True
    make_plots = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using Device: ", device)

    if make_plots:
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "figure.max_open_warning": 0,
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )

    for hidden_size in hidden_sizes:
        print("Processing: ", hidden_size)
        file_prefix = "{:06}".format(hidden_size)
        data_path = os.path.join(get_figure_dir(), "_".join([file_prefix, "data.pkl"]))
        if load and os.path.exists(data_path):
            history = torch.load(data_path)
            print("* Loaded existing")
        else:
            history = run(hidden_size, file_prefix, epochs=10)
            torch.save(history, data_path)
        print("* Final: ",
              history_list("running_acc", history)[-1],
              history_list("val_acc", history)[-1]
        )
        if make_plots:
            history_plots(history, file_prefix)
