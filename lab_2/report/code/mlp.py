def sgd_mlp(
    tr_data: Tensor, targets_tr: Tensor, num_epochs: int = 100, lr: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    W1 = torch.randn((4, 12), requires_grad=True)
    W2 = torch.randn((12, 3), requires_grad=True)
    b1 = torch.zeros(1, requires_grad=True)
    b2 = torch.zeros(1, requires_grad=True)

    for epoch in range(num_epochs):
        logits = mlp_func(tr_data, W1, W2, b1, b2)
        cross_entropy = torch.nn.functional.cross_entropy(
            logits, targets_tr, reduction="sum"
        )
        cross_entropy.backward()
        with torch.no_grad():
            W1 -= lr * W1.grad
            W2 -= lr * W2.grad
            b1 -= lr * b1.grad
            b2 -= lr * b2.grad
        for zv in [W1, W2, b1, b2]:
            zv.grad.zero_()

    return W1.detach(), W2.detach(), b1.detach(), b2.detach()