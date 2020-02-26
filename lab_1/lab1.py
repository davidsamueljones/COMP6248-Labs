import torch

from typing import Tuple
from torch import DoubleTensor, Tensor


def sgd_factorise(
    A: Tensor, rank: int, num_epochs: int = 1000, lr: float = 0.01
) -> Tuple[Tensor, Tensor]:
    (m, n) = A.shape
    U = torch.randn((m, rank))
    V = torch.randn((n, rank))
    for epoch in range(num_epochs):
        e = A - U @ V.t()
        for r in range(m):
            for c in range(n):
                e = A[r, c] - U[r] @ V[c].t()
                U[r] += lr * e * V[c]
                V[c] += lr * e * U[r]
    return U, V

def sgd_factorise_masked(
    A: Tensor, M: Tensor, rank: int, num_epochs: int = 1000, lr: float = 0.01
) -> Tuple[Tensor, Tensor]:
    (m, n) = A.shape
    U = torch.randn((m, rank))
    V = torch.randn((n, rank))
    for epoch in range(num_epochs):
        e = A - U @ V.t()
        for r in range(m):
            for c in range(n):
                if M[r, c]:
                    e = A[r, c] - U[r] @ V[c].t()
                    U[r] += lr * e * V[c]
                    V[c] += lr * e * U[r]
    return U, V

def svd_factorise(A: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    U, S, V = torch.svd(A)
    S[rank:] = 0
    return (U, torch.diag(S), V)


if __name__ == "__main__":
    # torch.manual_seed(0)
    # torch.set_default_dtype(torch.float64)

    A = torch.tensor(
        [
            [0.3374, 0.6005, 0.1735],  #
            [3.3359, 0.0492, 1.8374],  #
            [2.9407, 0.5301, 2.2620],  #
        ]
    )
    M = torch.tensor(
        [
            [1, 1, 1],  #
            [0, 1, 1],  #
            [1, 0, 1],  #
        ],
        dtype=bool
    )
    rank = 2
    (U, V) = sgd_factorise(A, rank)
    A_sgd_hat = U @ V.t()
    sgd_loss = torch.nn.functional.mse_loss(A_sgd_hat, A, reduction="sum")
    (U, S, V) = svd_factorise(A, rank)
    A_svd_hat = U @ S @ V.t()
    svd_loss = torch.nn.functional.mse_loss(A_svd_hat, A, reduction="sum")
    print("SGD Loss: {}".format(sgd_loss))
    print("SVD Loss: {}".format(svd_loss))
    (U, V) = sgd_factorise_masked(A, M, rank)
    A_sgd_masked_hat = U @ V.t()
    print("Masked Estimate: ")
    print(A_sgd_masked_hat.numpy())

