def gd_factorise_ad(
    A: Tensor, rank: int, num_epochs: int = 1000, lr: float = 0.01
) -> Tuple[Tensor, Tensor]:
    (m, n) = A.shape
    U = torch.rand((m, rank), requires_grad=True, dtype=torch.double)
    V = torch.rand((n, rank), requires_grad=True, dtype=torch.double)
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
