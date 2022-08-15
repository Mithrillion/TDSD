# evalutes the run time of TD-PCA SigDist on the SITS dataset

import os
import numpy as np
import pandas as pd
import time

import torch
import signatory

import sklearn.metrics as skm
from scipy.stats import mode

from utils.transforms import to_td_pca, rescale_path, unorm
from utils.functions import cross_knn_search

# hyperparameters
depth = 4
sig_mode = "words"
basepoint = True
tau, d, dims = 1, 18, 6
k = 10

DATASET_ROOT = "data/SITS_2006_NDVI_C"
dataset_list = sorted(
    [
        x
        for x in os.listdir(DATASET_ROOT)
        if not (x.endswith(".csv")) | (x.endswith(".xlsx"))
    ]
)

dataset_name = dataset_list[5]  # can choose an arbitrary fold - does not affect run time

train = pd.read_csv(
    f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TRAIN.csv", header=None
)
test = pd.read_csv(
    f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TEST.csv", header=None
)
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

x_max = np.quantile(X_train, 0.95)
x_min = np.quantile(X_train, 0.05)
X_train = (X_train - x_min) / (x_max - x_min)
X_test = (X_test - x_min) / (x_max - x_min)

t0 = time.time()

pc_train, pca = to_td_pca(torch.tensor(X_train).float()[..., None], dims, tau, d, backend="cuml")
pc_test, _ = to_td_pca(torch.tensor(X_test).float()[..., None], dims, tau, d, pca, backend="cuml")

sigs_train = signatory.logsignature(
    rescale_path(pc_train, depth),
    depth=depth,
    basepoint=basepoint,
    mode=sig_mode,
)
sigs_test = signatory.logsignature(
    rescale_path(pc_test, depth),
    depth=depth,
    basepoint=basepoint,
    mode=sig_mode,
)

D, I = cross_knn_search(unorm(sigs_test), unorm(sigs_train), k, "dot")

y_pred_sigdist, _ = mode(y_train[I], -1)
acc_sigdist = skm.accuracy_score(y_test, y_pred_sigdist)
f1_sigdist_weighted = skm.f1_score(y_test, y_pred_sigdist, average="weighted")

print(f"acc={acc_sigdist}")
print(f"f1={f1_sigdist_weighted}")

t1 = time.time()

total_n = t1 - t0
print(f"time={total_n}")