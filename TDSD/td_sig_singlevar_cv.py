import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import signatory

import sklearn.metrics as skm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from scipy.io import arff

sys.path.append("..")

from utils.transforms import to_td_pca, rescale_path, unorm
from utils.functions import cross_knn_search
import optuna

DATASET_ROOT = "data/Univariate_arff"
PCA_BACKEND = "cuml"
# NOTE: config here
no_pca = False  # whether to use only td or td-pca
td_pca = False  # true: alter both tau and d in hyperparameter search for td-pca; false: fix tau = 1
PRE_PCA = True  # if false, PCA will be re-performed for each validation batch, even if LOO - more accurate but can be very slow
dist_measure = "sig_cosine"  # can be "sig_ed", "sig_cosine" or "logsig_cosine"
random_seed = 7777


def load_data(dataset_name):
    train = arff.loadarff(f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TRAIN.arff")
    test = arff.loadarff(f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TEST.arff")

    label_enc = LabelEncoder()
    X_train = pd.DataFrame(train[0]).values[..., :-1].astype(np.float32)
    y_train = label_enc.fit_transform(pd.DataFrame(train[0]).values[..., -1])

    X_test = pd.DataFrame(test[0]).values[..., :-1].astype(np.float32)
    y_test = label_enc.transform(pd.DataFrame(test[0]).values[..., -1])

    x_max = np.quantile(X_train, 0.95)
    x_min = np.quantile(X_train, 0.05)
    X_train = (X_train - x_min) / (x_max - x_min)
    X_test = (X_test - x_min) / (x_max - x_min)

    X_train_tensor = torch.tensor(X_train).float()[..., None]
    X_test_tensor = torch.tensor(X_test).float()[..., None]

    return X_train_tensor, X_test_tensor, y_train, y_test


def sigdist_eval(
    X_train_tensor,
    X_test_tensor,
    y_train,
    y_test,
    tau,
    d,
    dims,
    depth,
    basepoint,
    use_logsig,
    dist_mode,
    sig_mode,
    pca_backend=PCA_BACKEND,
    pre_pca=False,
):

    if not pre_pca:
        td_X_train, pca = to_td_pca(
            X_train_tensor, dims, tau, d, backend=pca_backend, random_state=random_seed
        )
        td_X_test, _ = to_td_pca(
            X_test_tensor,
            dims,
            tau,
            d,
            pca,
            backend=pca_backend,
            random_state=random_seed,
        )
    else:
        td_X_train, td_X_test = X_train_tensor, X_test_tensor

    if use_logsig:
        sigs_train = signatory.logsignature(
            rescale_path(td_X_train, depth),
            depth=depth,
            basepoint=basepoint,
            mode=sig_mode,
        )
        sigs_test = signatory.logsignature(
            rescale_path(td_X_test, depth),
            depth=depth,
            basepoint=basepoint,
            mode=sig_mode,
        )
    else:
        sigs_train = signatory.signature(
            rescale_path(td_X_train, depth),
            depth=depth,
            basepoint=basepoint,
        )
        sigs_test = signatory.signature(
            rescale_path(td_X_test, depth),
            depth=depth,
            basepoint=basepoint,
        )

    if dist_mode == "ed":
        D, I = cross_knn_search(sigs_test, sigs_train, 1)
    elif dist_mode == "cosine":
        D, I = cross_knn_search(unorm(sigs_test), unorm(sigs_train), 1, "dot")
    else:
        raise ValueError("Invalid dist_mode!")

    y_pred_pca_sigdist = y_train[I.flatten()]
    acc = skm.accuracy_score(y_test, y_pred_pca_sigdist)
    f1 = skm.f1_score(y_test, y_pred_pca_sigdist, average="weighted")
    return acc, f1


if __name__ == "__main__":

    dataset_list = sorted(
        [
            x
            for x in os.listdir(DATASET_ROOT)
            if x
            not in [
                "ShakeGestureWiimoteZ",
                "PLAID",
                "GesturePebbleZ1",
                "AllGestureWiimoteY",
                "AllGestureWiimoteX",
                "GesturePebbleZ2",
                "PickupGestureWiimoteZ",
                "AllGestureWiimoteZ",
                "DodgerLoopDay",
                "DodgerLoopWeekend",
                "DodgerLoopGame",
                "Pictures",
                "GestureMidAirD1",
                "GestureMidAirD2",
                "GestureMidAirD3",
            ]
            and not x.endswith(".csv")
            and not x.endswith(".xlsx")
            and not x.startswith(".~")
        ]
    )

    if no_pca:
        td_pca, PRE_PCA = False, False

    results_df = pd.DataFrame()
    skipped = []
    error = []
    oom = []
    for dataset_name in tqdm(dataset_list):
        X_train_tensor, X_test_tensor, y_train, y_test = load_data(dataset_name)
        max_d = int(X_train_tensor.shape[1] * 0.4)
        n_ch = X_train_tensor.shape[-1]

        if len(X_train_tensor) >= 200:
            n_folds = 10
            loo = False
        else:
            loo = True

        if dist_measure == "sig_ed":
            fixed_params = dict(use_logsig=False, dist_mode="ed", sig_mode="words")
        elif dist_measure == "sig_cosine":
            fixed_params = dict(use_logsig=False, dist_mode="cosine", sig_mode="words")
        elif dist_measure == "logsig_cosine":
            fixed_params = dict(use_logsig=True, dist_mode="cosine", sig_mode="words")

        cv = (
            LeaveOneOut()
            if loo
            else StratifiedKFold(n_folds, random_state=random_seed, shuffle=True)
        )

        def objective(trial):
            if no_pca:
                # only for no pca mode
                d = trial.suggest_int("d", 2, 7)
                tau = trial.suggest_int("tau", 1, max_d // d)
            elif td_pca:
                d = trial.suggest_int("d", 2, min(max_d, 32))
                tau = trial.suggest_int("tau", 1, max_d // d)
                dims = trial.suggest_int("dims", 2, min(d, 7))
            else:
                d = trial.suggest_int("d", 2, max_d)
                dims = trial.suggest_int("dims", 2, max(2, min(d, 7)))

            depth = trial.suggest_int("depth", 3, 4)
            basepoint = trial.suggest_categorical("basepoint", [True, False])

            acc, f1 = sigdist_eval(
                X_train_tensor,
                X_test_tensor,
                y_train,
                y_test,
                tau=tau if (no_pca | td_pca) else 1,
                d=d,
                dims=None if no_pca else dims,
                depth=depth,
                basepoint=basepoint,
                **fixed_params,
            )

            return -acc, -f1

        def cv_objective(trial):
            if no_pca:
                # only for no pca mode
                d = trial.suggest_int("d", 2, 7)
                tau = trial.suggest_int("tau", 1, max_d // d)
            elif td_pca:
                d = trial.suggest_int("d", 2, min(max_d, 32))
                tau = trial.suggest_int("tau", 1, max_d // d)
                dims = trial.suggest_int("dims", 2, min(d, 7))
            else:
                d = trial.suggest_int("d", 2, max_d)
                dims = trial.suggest_int("dims", 2, max(2, min(d, 7)))
                tau = tau if (no_pca | td_pca) else 1

            depth = trial.suggest_int("depth", 3, 4)
            basepoint = trial.suggest_categorical("basepoint", [True, False])

            if PRE_PCA:
                td_X_train, _ = to_td_pca(
                    X_train_tensor,
                    dims,
                    tau,
                    d,
                    backend=PCA_BACKEND,
                    random_state=random_seed,
                )

            acc_list = []
            for train_index, test_index in cv.split(X_train_tensor, y_train):
                # if leave one out
                if loo:
                    test_index = np.repeat(test_index, 2)

                if PRE_PCA:
                    train_X, val_X = td_X_train[train_index], td_X_train[test_index]
                else:
                    train_X, val_X = (
                        X_train_tensor[train_index],
                        X_train_tensor[test_index],
                    )
                train_y, val_y = y_train[train_index], y_train[test_index]
                acc, f1 = sigdist_eval(
                    train_X,
                    val_X,
                    train_y,
                    val_y,
                    tau=tau,
                    d=d,
                    dims=None if no_pca else dims,
                    depth=depth,
                    basepoint=basepoint,
                    pre_pca=PRE_PCA,
                    **fixed_params,
                )
                acc_list += [acc]

            return -np.mean(acc_list)

        def objective_reached_callback(study, trial):
            if trial.value == -1:
                study.stop()

        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(sampler=sampler)
        try:
            study.optimize(
                cv_objective, n_trials=50, callbacks=[objective_reached_callback]
            )
        except ValueError:
            skipped += [dataset_name]
            continue
        except RuntimeError:
            error += [dataset_name]
            continue
        except MemoryError:
            oom += [dataset_name]
            continue

        test_acc, test_f1 = objective(study.best_trial)
        results = {
            "dataset": dataset_name,
            "length": X_train_tensor.shape[1],
            "classes": len(np.unique(y_train)),
            "cv": "LOO" if loo else "10-Fold",
            "PCA": "N" if no_pca else ("TD" if td_pca else "D"),
            "dist_measure": dist_measure,
            "train_acc": -study.best_value,
            "test_acc": -test_acc,
            "test_f1": -test_f1,
        }
        results = results | study.best_params

        results_df = results_df.append(pd.Series(results), ignore_index=True)

    results_df["length"] = results_df["length"].astype(int)
    results_df["classes"] = results_df["classes"].astype(int)
    results_df["d"] = results_df["d"].astype(int)
    if "tau" in results_df.columns:
        results_df["tau"] = results_df["tau"].astype(int)
    results_df["depth"] = results_df["depth"].astype(int)
    results_df["basepoint"] = results_df["basepoint"].astype(bool)
    if "dims" in results_df.columns:
        results_df["dims"] = results_df["dims"].astype(int)

    results_df.to_csv("results/pca_cos_seeded.csv")
