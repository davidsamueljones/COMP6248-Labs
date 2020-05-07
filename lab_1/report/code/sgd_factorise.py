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
