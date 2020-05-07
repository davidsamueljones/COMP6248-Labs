def svd_factorise(A: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    U, S, V = torch.svd(A)
    S[rank:] = 0
    return (U, torch.diag(S), V)