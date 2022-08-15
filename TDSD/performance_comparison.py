import os
import numpy as np
import pandas as pd
import time
from dtaidistance import dtw
import sklearn.metrics as skm
from scipy.io import arff
from tqdm import tqdm
import signatory
import torch

from utils.transforms import to_td_pca, rescale_path, unorm
from utils.functions import cross_knn_search

import holoviews as hv

hv.extension("bokeh")

dtw_n_iters = 1
sig_n_iters = 1

if __name__ == "__main__":

    ########################
    # DTW Run Time
    ########################

    DATASET_ROOT = "data/Univariate_arff"
    dataset_list = [
        x
        for x in os.listdir(DATASET_ROOT)
        if not (x.endswith(".csv")) | (x.endswith(".xlsx"))
    ]

    dataset_name = "Crop"

    train = arff.loadarff(f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TRAIN.arff")
    test = arff.loadarff(f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TEST.arff")

    dat_train = pd.DataFrame(train[0])
    dat_test = pd.DataFrame(test[0])

    y_train = dat_train["target"].astype(int)
    y_test = dat_test["target"].astype(int)

    X_train = dat_train.iloc[:, :-1].values
    scale_min, scale_max = X_train.min(), X_train.max()
    X_train = (X_train - scale_min) / (scale_max - scale_min)
    X_test = dat_test.iloc[:, :-1].values
    X_test = (X_test - scale_min) / (scale_max - scale_min)

    t0 = time.time()

    for i in range(dtw_n_iters):
        combined = np.concatenate([X_test, X_train], axis=0)
        cross_dtw = dtw.distance_matrix_fast(
            combined,
            use_pruning=True,
            window=4,
            block=((0, len(X_test)), (len(X_test), len(combined))),
        )[: len(X_test), len(X_test) :]
        y_pred_dtw = y_train[np.argmin(cross_dtw, -1)]
        acc_dtw = skm.accuracy_score(y_test, y_pred_dtw)
        f1_dtw = skm.f1_score(y_test, y_pred_dtw, average="weighted")

    t1 = time.time()

    print(f"DTW ACC: {acc_dtw}")
    print(f"DTW F1: {f1_dtw}")

    total_n = t1 - t0
    print(f"DTW runtime: {total_n / dtw_n_iters}")
    # 26.94

    ########################
    # SigDist Run Time
    ########################

    depth = 4
    sig_mode = "words"
    basepoint = True
    tau, d, dims = 1, 18, 6
    k = 10

    t0 = time.time()

    for i in range(sig_n_iters * 10):
        pc_train, pca = to_td_pca(
            torch.tensor(X_train).float()[..., None], dims, tau, d, backend="cuml"
        )
        pc_test, _ = to_td_pca(
            torch.tensor(X_test).float()[..., None], dims, tau, d, pca, backend="cuml"
        )

        sigs_train = signatory.signature(
            rescale_path(pc_train, depth),
            depth=depth,
            basepoint=basepoint,
            # mode=sig_mode,
        )
        sigs_test = signatory.signature(
            rescale_path(pc_test, depth),
            depth=depth,
            basepoint=basepoint,
            # mode=sig_mode,
        )

        D, I = cross_knn_search(unorm(sigs_test), unorm(sigs_train), 1, "dot")

        y_pred_sigdist = y_train[I.flatten().numpy()]
        acc_sigdist = skm.accuracy_score(y_test, y_pred_sigdist)
        f1_sigdist_weighted = skm.f1_score(y_test, y_pred_sigdist, average="weighted")

    t1 = time.time()

    print(f"SigDist ACC: {acc_sigdist}")
    print(f"SigDist F1: {f1_sigdist_weighted}")

    total_n = t1 - t0
    print(f"SigDist Runtime: {total_n / sig_n_iters / 10}")

    ########################
    # DTW #samples multiplied
    ########################

    timings = []

    for factor in tqdm(range(1, 6)):
        combined = np.concatenate([X_test, X_train], axis=0)
        combined = np.repeat(combined, factor, 0)

        t0 = time.time()

        for i in range(dtw_n_iters):
            cross_dtw = dtw.distance_matrix_fast(
                combined,
                use_pruning=True,
                window=4,
                block=(
                    (0, factor * len(X_test)),
                    (factor * len(X_test), len(combined)),
                ),
                compact=True,
            )

        t1 = time.time()

        total_n = t1 - t0
        timings += [total_n / dtw_n_iters]

    ########################
    # SigDist #samples multiplied
    ########################

    # NOTE: SigDist
    depth = 4
    sig_mode = "words"
    basepoint = True
    tau, d, dims = 1, 18, 6
    k = 10
    sig_timings = []

    for factor in tqdm(range(1, 10)):

        t0 = time.time()

        for i in range(sig_n_iters):
            pc_train, pca = to_td_pca(
                torch.tensor(np.repeat(X_train, factor, 0)).float()[..., None],
                dims,
                tau,
                d,
                backend="cuml",
            )
            pc_test, _ = to_td_pca(
                torch.tensor(np.repeat(X_test, factor, 0)).float()[..., None],
                dims,
                tau,
                d,
                pca,
                backend="cuml",
            )

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

            D, I = cross_knn_search(unorm(sigs_test), unorm(sigs_train), 1, "dot")

        t1 = time.time()

        total_n = t1 - t0
        sig_timings += [total_n / sig_n_iters]

    ########################
    # DTW len multiplied
    ########################

    dataset_name = "StarLightCurves"

    train = arff.loadarff(f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TRAIN.arff")
    test = arff.loadarff(f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TEST.arff")

    dat_train = pd.DataFrame(train[0])
    dat_test = pd.DataFrame(test[0])

    y_train = dat_train["target"].astype(int)
    y_test = dat_test["target"].astype(int)

    X_train = dat_train.iloc[:, :-1].values
    scale_min, scale_max = X_train.min(), X_train.max()
    X_train = (X_train - scale_min) / (scale_max - scale_min)
    X_test = dat_test.iloc[:, :-1].values
    X_test = (X_test - scale_min) / (scale_max - scale_min)

    len_timings = []

    for factor in tqdm(range(1, 5)):
        combined = np.concatenate([X_test, X_train], axis=0)
        combined = np.repeat(combined, factor, 1)

        t0 = time.time()

        for i in range(dtw_n_iters):
            cross_dtw = dtw.distance_matrix_fast(
                combined,
                use_pruning=True,
                window=16,
                block=((0, len(X_test)), (len(X_test), len(combined))),
                compact=True,
            )

        t1 = time.time()

        total_n = t1 - t0
        len_timings += [total_n / dtw_n_iters]

    ########################
    # SigDist len multiplied
    ########################

    depth = 4
    sig_mode = "words"
    basepoint = True
    tau, d, dims = 1, 18, 6
    k = 10
    len_sig_timings = []

    for factor in tqdm(range(1, 7)):

        t0 = time.time()

        for i in range(sig_n_iters):
            pc_train, pca = to_td_pca(
                torch.tensor(np.repeat(X_train, factor, 1)).float()[..., None],
                dims,
                tau,
                d,
                backend="cuml",
            )
            pc_test, _ = to_td_pca(
                torch.tensor(np.repeat(X_test, factor, 1)).float()[..., None],
                dims,
                tau,
                d,
                pca,
                backend="cuml",
            )

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

            D, I = cross_knn_search(unorm(sigs_test), unorm(sigs_train), 1, "dot")

        t1 = time.time()

        total_n = t1 - t0
        len_sig_timings += [total_n / sig_n_iters]

    ########################
    # Tempo #samples & len multiplied
    ########################

    len_tempo_timings = [1.744, 3.995, 6.107, 8.335, 11.204, 14.350]
    tempo_timings = np.array([[1, 2, 4, 8], [0.630, 2.324, 10.907, 51.069]])

    ########################
    # Collecting results
    ########################

    g1 = (
        hv.Curve(
            (np.arange(1, 6), np.log(1 + np.array(timings))),
            "multiples of dataset size",
            "log(10) runtime (s)",
            label="DTW",
        ).opts(width=500, height=400, fontscale=2)
        * hv.Curve(
            (np.arange(1, 9), np.log(1 + np.array(sig_timings[:-1]))), label="TDSD"
        )
        * hv.Curve(
            (tempo_timings[0, :], np.log(1 + tempo_timings[1, :])), label="Tempo"
        )
    )

    g2 = (
        hv.Curve(
            (np.arange(1, 5), np.log(1 + np.array(len_timings))),
            "multiples of sequence length",
            "log(10) runtime (s)",
            label="DTW",
        ).opts(width=500, height=400, fontscale=2)
        * hv.Curve(
            (np.arange(1, 7), np.log(1 + np.array(len_sig_timings))), label="TDSD"
        )
        * hv.Curve(
            (np.arange(1, 7), np.log(1 + np.array(len_tempo_timings))), label="Tempo"
        )
    )

    pl = (g1 + g2).opts(shared_axes=False)

    hv.save(pl, "results/comp.png")
