import signatory
import torch
import warnings

import math
from sklearn.decomposition import PCA

try:
    import cuml
    import cupy as cp

    NO_CUML = False
except ImportError:
    NO_CUML = True
    warnings.warn("cuml import failed, falling back to sklearn!")

# adapted from https://github.com/jambo6/generalised-signature-method/blob/master/src/features/signatures/transforms.py
def to_td(X, tau=1, d=2, pad=None, time_last=True):
    # last dimension assumed to be time
    if not time_last:
        # then it must be (batch, time, channel)
        X = X.transpose(-2, -1)
    if d < 2:
        raise ValueError("TD dimension must be at least 2!")
    if pad == "last":
        X_pad = torch.cat(
            [X, torch.repeat_interleave(X[..., [-1]], tau * d, -1)], dim=-1
        )
    elif pad == "first":
        X_pad = torch.cat(
            [X, torch.repeat_interleave(X[..., [0]], tau * d, -1)], dim=-1
        )
    else:
        X_pad = X
    return torch.stack(
        [X_pad[..., i * tau : -(d - i) * tau] for i in range(d)], dim=-1
    ).float()


def time_aug(X, t_range=(0, 1)):
    # assume (batch, time, channel)
    t_len = X.shape[1]
    time_axis = torch.linspace(*t_range, t_len)
    time_component = torch.repeat_interleave(time_axis[None, :, None], len(X), dim=0)
    return torch.cat([time_component, X], dim=-1)


def znorm(x, dim=-1):
    return (x - x.mean(dim, keepdim=True)) / (1e-8 + x.std(dim, keepdim=True))


def unorm(x, dim=-1):
    return x / (1e-8 + torch.norm(x, dim=dim, keepdim=True))


def to_td_pca(
    X,
    dims=2,
    tau=1,
    d=10,
    pretrained_pca=None,
    pad=None,
    backend="sklearn",
    random_state=None,
):
    # (batch, time, channel) -> (batch, channel, time, delays)
    # -> (batch, time, channel * delays)
    if d == 1:
        X_td = X
    else:
        X_td = to_td(X, tau, d, pad, time_last=False).transpose(1, 2).flatten(2, 3)
    batch_dims = X_td.shape[:-1]
    if dims is not None and dims != 0:
        if pretrained_pca:
            pca = pretrained_pca
            if backend == "cuml" and not NO_CUML:
                X_in = X_td.reshape(-1, X_td.shape[-1])
                X_in = cp.asarray(X_in)
                Z = pca.transform(X_in)
            elif backend == "sklearn":
                Z = pca.transform(X_td.reshape(-1, X_td.shape[-1]))
            else:
                raise ValueError("Invalid backend option!")
        else:
            if backend == "cuml" and not NO_CUML:
                pca = cuml.PCA(
                    n_components=dims, output_type="numpy", random_state=random_state
                )
                X_in = X_td.reshape(-1, X_td.shape[-1])
                X_in = cp.asarray(X_in)
                Z = pca.fit_transform(X_in)
            elif backend == "sklearn":
                pca = PCA(n_components=dims, random_state=random_state)
                Z = pca.fit_transform(X_td.reshape(-1, X_td.shape[-1]))
            else:
                raise ValueError("Invalid backend option!")
        return torch.tensor(Z).float().view(*batch_dims, -1), pca
    else:
        # dims is None or 0 -> do not perform pca
        return X_td, None


def to_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, (channel)]
    X_repeat = X.repeat_interleave(2, dim=1)

    # Split out lead and lag
    lead = X_repeat[:, 1:, :]
    lag = X_repeat[:, :-1, :]

    # Combine
    X_leadlag = torch.cat((lead, lag), 2)

    return X_leadlag


def to_nd_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, channel]
    n_chs = X.shape[-1]
    X_repeat = X.repeat_interleave(n_chs * 2, dim=1)
    streams = []
    for i in range(2):
        for j in range(n_chs):
            streams += [
                X_repeat[
                    :, 2 * n_chs - 2 * i - j - 1 : (X_repeat.shape[1] - 2 * i - j), j
                ]
            ]
    X_leadlag = torch.stack(streams, 2)
    return X_leadlag


# NOTE: adapted from https://github.com/jambo6/generalised-signature-method/blob/master/src/models/rescaling.py
def rescale_path(path, depth):
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path


def rescale_signature(signature, channels, depth):
    sigtensor_channels = signature.size(-1)
    if signatory.signature_channels(channels, depth) != sigtensor_channels:
        raise ValueError(
            "Given a sigtensor with {} channels, a path with {} channels and a depth of {}, which are "
            "not consistent.".format(sigtensor_channels, channels, depth)
        )

    end = 0
    term_length = 1
    val = 1
    terms = []
    for d in range(1, depth + 1):
        start = end
        term_length *= channels
        end = start + term_length

        val *= d

        terms.append(signature[..., start:end] * val)

    return torch.cat(terms, dim=-1)
