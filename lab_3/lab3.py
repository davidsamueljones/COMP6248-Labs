import torch
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from torch import Tensor
from nptyping import Array
from typing import Tuple


def rastrigin(X: Tensor, A: int = 1):
    d = len(X.shape)
    R = A * d
    for x in X:
        R += x ** 2 - A * torch.cos(2 * math.pi * x)
    return R


def get_data():
    x_min, x_max, x_step = -5, 5, 0.05
    y_min, y_max, y_step = -5, 5, 0.05
    X, Y = np.meshgrid(
        np.arange(x_min, x_max + x_step, x_step),
        np.arange(y_min, y_max + y_step, y_step),
    )
    Z = rastrigin(torch.tensor([X, Y])).numpy()
    return X, Y, Z


def hinge_loss(y_pred, y_true):
    hinge_loss = 1 - torch.mul(y_pred, y_true)
    hinge_loss[hinge_loss < 0] = 0
    return torch.mean(hinge_loss)


def svm(x, w, b):
    h = (w * x).sum(1) + b
    return h


def optimise(
    optimiser: optim.Optimizer, epochs: int = 100
) -> Tuple[Array[float], Array[float]]:
    # Extract only parameter so it can be referenced
    p = optimiser.param_groups[0]["params"][0]
    # Track path and losses
    path = np.empty((2, epochs + 1))
    path[:, 0] = p.data.numpy().flatten()
    losses = np.empty((epochs + 1))

    # Run optimiser
    for i in range(epochs):
        optimiser.zero_grad()
        output = rastrigin(p)
        output.backward()
        optimiser.step()
        path[:, i + 1] = p.data.numpy().flatten()
        losses[i] = output.data.numpy()
    losses[-1] = rastrigin(p).data.numpy()
    return path, losses


def optimise_svm(optimiser: optim.Optimizer, dataloader, epochs: int = 100):
    # Extract parameters so they can be referenced
    w = optimiser.param_groups[0]["params"][0]
    b = optimiser.param_groups[0]["params"][1]
    # Track path and losses
    w_path = np.empty((w.shape[1], epochs + 1))
    w_path[:, 0] = w.data.numpy().flatten()
    b_path = np.empty((b.shape[0], epochs + 1))
    b_path[:, 0] = b.data.numpy().flatten()
    losses = np.empty((epochs + 1))

    # Run optimiser
    for i in range(epochs):
        for (data, target) in dataloader:
            optimiser.zero_grad()
            pred = svm(data, w, b)
            loss = hinge_loss(pred, target)
            loss.backward()
            optimiser.step()
            w_path[:, i + 1] = b.data.numpy().flatten()
            b_path[:, i + 1] = b.data.numpy().flatten()
        # losses[i] = output.data.numpy()
    losses[-1] = losses[-2] # hack the last value to duplicate 2nd to last
    return w_path, b_path, losses


def ex_1():
    X, Y, Z = get_data()
    p_base = torch.tensor([[5.0], [5.0]], requires_grad=True)
    p = lambda: p_base.clone().detach().requires_grad_(True)
    lr = 0.01
    sgd_path, sgd_losses = optimise(optim.SGD([p()], lr=lr))
    sgd_mom_path, sgd_mom_losses = optimise(optim.SGD([p()], lr=lr, momentum=0.9))
    adagrad_path, adagrad_losses = optimise(optim.Adagrad([p()], lr=lr))
    adam_path, adam_losses = optimise(optim.Adam([p()], lr=lr, betas=[0.7, 0.999]))

    fig = plt.figure()
    plt.plot(sgd_losses, label="SGD")
    plt.plot(sgd_mom_losses, label="SGD + Momentum")
    plt.plot(adagrad_losses, label="Adagrad")
    plt.plot(adam_losses, label="Adam")
    plt.legend()

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection="3d", elev=50, azim=-50)
    ax.plot_surface(
        X,
        Y,
        Z,
        norm=LogNorm(),
        rstride=1,
        cstride=1,
        edgecolor="none",
        alpha=0.75,
        cmap=plt.cm.gray,
    )
    ax.plot(
        sgd_mom_path[0],
        sgd_mom_path[1],
        zs=sgd_mom_losses,
        color="yellow",
        label="SGD + Momentum",
        linewidth=2,
    )
    ax.plot(
        sgd_path[0], sgd_path[1], zs=sgd_losses, color="red", label="SGD", linewidth=2
    )
    ax.plot(
        adam_path[0],
        adam_path[1],
        zs=adam_losses,
        color="green",
        label="Adam",
        linewidth=2,
    )
    ax.plot(
        adagrad_path[0],
        adagrad_path[1],
        zs=adagrad_losses,
        color="blue",
        label="Adagrad",
        linewidth=2,
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")

    ax.set_xlim((np.amin(X), np.amax(X)))
    ax.set_ylim((np.amin(Y), np.amax(Y)))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(X, Y, Z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)

    ax.plot(
        sgd_mom_path[0],
        sgd_mom_path[1],
        color="yellow",
        label="SGD + Momentum",
        linewidth=2,
    )
    ax.plot(sgd_path[0], sgd_path[1], color="red", label="SGD", linewidth=2)
    ax.plot(adam_path[0], adam_path[1], color="green", label="Adam", linewidth=2)
    ax.plot(
        adagrad_path[0], adagrad_path[1], color="blue", label="Adagrad", linewidth=2
    )

    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.set_xlim((np.amin(X), np.amax(X)))
    ax.set_ylim((np.amin(Y), np.amax(Y)))


if __name__ == "__main__":
    ex_1()

    # Load data
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    df = df.sample(frac=1)
    df = df[df[4].isin(["Iris-virginica", "Iris-versicolor"])]
    # Add label indices column
    mapping = {k: v for v, k in enumerate(df[4].unique())}
    df[5] = (2 * df[4].map(mapping)) - 1
    # Normalise data
    alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
    alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)
    # Create datasets
    targets_tr = torch.tensor(df.iloc[:75, 5].values, dtype=torch.long)
    targets_va = torch.tensor(df.iloc[75:, 5].values, dtype=torch.long)
    data_tr = alldata[:75]
    data_va = alldata[75:]

    dataset = torch.utils.data.TensorDataset(data_tr, targets_tr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True)

    fig = plt.figure(figsize=(5, 5))

    w = torch.randn(1, 4, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    opt = optim.SGD([w, b], lr=0.01, weight_decay=0.0001)
    sgd_w_path, sgd_b_path, sgd_losses = optimise_svm(opt, dataloader)

    w = torch.randn(1, 4, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    opt = optim.Adam([w, b], lr=0.01, weight_decay=0.0001)
    adam_w_path, adam_b_path, adam_losses = optimise_svm(opt, dataloader)


    print("SGD")
    w = torch.tensor(sgd_w_path[:, -1])
    b = torch.tensor(sgd_b_path[:, -1])

    pred = svm(data_tr, w, b).detach().numpy()
    pred[pred > 0] = 1
    pred[pred < 0] = -1
    labels = targets_tr.detach().numpy()
    accuracy = (labels == pred).sum() / labels.shape[0]
    print("Training: ", accuracy)

    pred = svm(data_va, w, b).detach().numpy()
    pred[pred > 0] = 1
    pred[pred < 0] = -1
    labels = targets_va.detach().numpy()
    accuracy = (labels == pred).sum() / labels.shape[0]
    print("Verification: ", accuracy)

    w = torch.tensor(adam_w_path[:, -1])
    b = torch.tensor(adam_b_path[:, -1])

    print("Adam")
    pred = svm(data_tr, w, b).detach().numpy()
    pred[pred > 0] = 1
    pred[pred < 0] = -1
    labels = targets_tr.detach().numpy()
    accuracy = (labels == pred).sum() / labels.shape[0]
    print("Training: ", accuracy)

    pred = svm(data_va, w, b).detach().numpy()
    pred[pred > 0] = 1
    pred[pred < 0] = -1
    labels = targets_va.detach().numpy()
    accuracy = (labels == pred).sum() / labels.shape[0]
    print("Verification: ", accuracy)

    plt.show()