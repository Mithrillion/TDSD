import torch
import typing
import pykeops.torch as ktorch


#############################################
# kNN searches
#############################################


def knn_search(
    feats: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.tensor, torch.tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def cross_knn_search(
    A: torch.Tensor, B: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(A[:, None, :])
    X_j = ktorch.LazyTensor(B[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I
