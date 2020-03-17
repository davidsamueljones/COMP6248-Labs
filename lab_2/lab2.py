import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from torch import DoubleTensor, Tensor
from matplotlib import rc


def sgd_factorise_ad(
    A: Tensor, rank: int, num_epochs: int = 1000, lr: float = 0.01
) -> Tuple[Tensor, Tensor]:
    (m, n) = A.shape
    U = torch.randn((m, rank), requires_grad=True, dtype=torch.double)
    V = torch.randn((n, rank), requires_grad=True, dtype=torch.double)
    for epoch in range(num_epochs):
        A_sgd_hat = U @ V.t()
        loss = torch.nn.functional.mse_loss(A_sgd_hat, A, reduction="sum")
        loss.backward(torch.ones(loss.shape))
        with torch.no_grad():
            U -= lr * U.grad
            V -= lr * V.grad
        U.grad.zero_()
        V.grad.zero_()
    return U.detach(), V.detach()


def svd_factorise(A: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    U, S, V = torch.svd(A)
    S[rank:] = 0
    return (U, torch.diag(S), V)


def lab_1_test():
    A = torch.tensor(
        [
            [0.3374, 0.6005, 0.1735],  #
            [3.3359, 0.0492, 1.8374],  #
            [2.9407, 0.5301, 2.2620],  #
        ],
        dtype=torch.double,
    )
    rank = 2
    (U, V) = sgd_factorise_ad(A, rank)
    A_sgd_hat = U @ V.t()
    sgd_loss = torch.nn.functional.mse_loss(A_sgd_hat, A, reduction="sum")
    print(A_sgd_hat)
    print("SGD Loss: {}".format(sgd_loss))

def ex_1():
    # Load data
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    data = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
    labels = df.iloc[:, 4].values
    data = data - data.mean(dim=0)
    # Configure exercise
    D = data
    rank = 2
    # Get SGD and SVD Result
    (U_hat, V) = sgd_factorise_ad(D, rank)
    D_sgd_hat = U_hat @ V.t()
    sgd_loss = torch.nn.functional.mse_loss(D_sgd_hat, D, reduction="sum")
    (U, S, V) = svd_factorise(D, rank)
    D_svd_hat = U @ S @ V.t()
    svd_loss = torch.nn.functional.mse_loss(D_svd_hat, D, reduction="sum")
    print("SGD Loss: {}".format(sgd_loss))
    print("SVD Loss: {}".format(svd_loss))
    # Plot
    classes = np.unique(labels).tolist()
    class_cnt = len(classes)
    colors = ["red", "green", "blue", "purple", "yellow"]
    markers = ["o", "X", "D", "^", "*"]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for ui, U_data in enumerate([U, U_hat]):
        for ci, c in enumerate(classes):
            cd = U_data[labels == c]
            kwargs = {
                "alpha": 0.5,
                "color": colors[ci],
                "label": c,
                "marker": markers[ci],
            }
            ax[ui].scatter(cd[:, 0], cd[:, 1], **kwargs)
    ax[1].legend(loc=1)

    # https://towardsdatascience.com/pca-vs-autoencoders-1ba08362f450
    # https://www.cs.toronto.edu/~urtasun/courses/CSC411/14_pca.pdf

def mlp_func(data: Tensor, W1: Tensor, W2:Tensor, b1: Tensor, b2:Tensor) -> Tensor:
    return torch.relu(data @ W1 + b1) @ W2 + b2

def sgd_mlp(tr_data: Tensor, targets_tr: Tensor, num_epochs: int = 100, lr: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    W1 = torch.randn((4, 12), requires_grad=True)
    W2 = torch.randn((12, 3), requires_grad=True)
    b1 = torch.zeros(1, requires_grad=True)
    b2 = torch.zeros(1, requires_grad=True)

    for epoch in range(num_epochs):
        logits = mlp_func(tr_data, W1, W2, b1, b2)
        cross_entropy = torch.nn.functional.cross_entropy(logits, targets_tr, reduction="sum")
        cross_entropy.backward()
        with torch.no_grad():
            W1 -= lr * W1.grad
            W2 -= lr * W2.grad
            b1 -= lr * b1.grad
            b2 -= lr * b2.grad
        for zv in [W1, W2, b1, b2]:
            zv.grad.zero_()

    return W1.detach(), W2.detach(), b1.detach(), b2.detach()

def run_mlp(data: Tensor, targets: Tensor, W1: Tensor, W2:Tensor, b1: Tensor, b2:Tensor) -> Tensor:
    logits = mlp_func(data, W1, W2, b1, b2)
    predictions = np.argmax(logits.numpy(), axis=1)
    for c in sorted(np.unique(targets)):
        cvs = np.argwhere(np.equal(targets, c))
        matches = np.equal(predictions[cvs][0], targets[cvs][0])
        accuracy = matches.sum().item() / matches.shape[0]
        print("C{} : {}".format(c, accuracy))
    matches = np.equal(predictions, targets)
    accuracy = matches.sum().item() / matches.shape[0]
    return predictions, logits, accuracy

def ex_2():
    # Load data
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    df = df.sample(frac=1)  # shuffle
    # Add label indices column
    labels = df[4].to_numpy()
    mapping = {k: v for v, k in enumerate(df[4].unique())}
    df[5] = df[4].map(mapping)

    # Normalise data
    alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
    alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)
    # Create datasets
    targets_tr = torch.tensor(df.iloc[:100, 5].values, dtype=torch.long)
    targets_va = torch.tensor(df.iloc[100:, 5].values, dtype=torch.long)
    data_tr = alldata[:100]
    data_va = alldata[100:]

    W1, W2, b1, b2 = sgd_mlp(data_tr, targets_tr)
    tr_ps, tr_logits, tr_acc = run_mlp(data_tr, targets_tr, W1, W2, b1, b2)
    va_ps, va_logits, va_acc = run_mlp(data_va, targets_va, W1, W2, b1, b2)
    print(tr_acc * 100)
    print(va_acc * 100)

    # classes = np.unique(labels).tolist()
    # class_cnt = len(classes)
    # colors = ["red", "green", "blue", "purple", "yellow"]
    # markers = ["o", "X", "D", "^", "*"]
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # for ui, U_data in enumerate([(tr_ps), va_ps]):
    #     for ci, c in enumerate(classes):
    #         cd = U_data[labels == c]
    #         kwargs = {
    #             "alpha": 0.5,
    #             "color": colors[ci],
    #             "label": c,
    #             "marker": markers[ci],
    #         }
    #         ax[ui].scatter(cd[:, 0], cd[:, 1], **kwargs)
    # ax.legend(loc=1)


if __name__ == "__main__":
    seed = 0
    if seed is None:
        seed = torch.seed()
    else:
        torch.manual_seed(seed)
    print("Seed: {}".format(seed))
    rc("font", **{"family": "DejaVu Sans", "sans-serif": ["Helvetica"]})
    rc("text", usetex=True)

    if False:
        print("\nLab 1 Verification")
        lab_1_test()

    print("\n[1] Matrix Factorisation")
    ex_1()

    print("\n[2] MLP")
    ex_2()

    plt.show()