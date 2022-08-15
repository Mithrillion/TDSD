# this script evaluates the baseline performance (accuracy and f1) of dtaidistance's implementation
# of DTW for 1NN classification on the UCR Archive, using optimal warping window sizes from the UCR Archive metadata

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from dtaidistance import dtw

import sklearn.metrics as skm
from sklearn.preprocessing import LabelEncoder

from scipy.io import arff

if __name__ == "__main__":

    DATASET_ROOT = "data/Univariate_arff"
    dataset_list = sorted(
        [
            x
            for x in os.listdir(DATASET_ROOT)
            if not (x.endswith(".csv")) | (x.endswith(".xlsx"))
            and x not in ["Descriptions", "Images", "Pictures"]
        ]
    )

    windows = pd.read_csv("data/opt_windows.csv").set_index("Name")

    results_dict = dict()
    for dataset_name in tqdm(sorted(dataset_list)):
        train = arff.loadarff(
            f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TRAIN.arff"
        )
        test = arff.loadarff(f"{DATASET_ROOT}/{dataset_name}/{dataset_name}_TEST.arff")

        label_enc = LabelEncoder()
        X_train = pd.DataFrame(train[0]).values[..., :-1].astype(np.double)
        y_train = label_enc.fit_transform(pd.DataFrame(train[0]).values[..., -1])

        X_test = pd.DataFrame(test[0]).values[..., :-1].astype(np.double)
        y_test = label_enc.transform(pd.DataFrame(test[0]).values[..., -1])

        X_train_tensor = torch.tensor(X_train).float().contiguous()
        X_test_tensor = torch.tensor(X_test).float().contiguous()

        window = windows.loc[dataset_name, "window"] + 1
        combined = np.concatenate([X_test, X_train], axis=0)
        cross_dtw = dtw.distance_matrix_fast(
            combined,
            use_pruning=True,
            window=window,
            block=((0, len(X_test)), (len(X_test), len(combined))),
        )[: len(X_test), len(X_test) :]
        y_pred_dtw = y_train[np.argmin(cross_dtw, -1)]
        acc_dtw = skm.accuracy_score(y_test, y_pred_dtw)
        f1_dtw_weighted = skm.f1_score(y_test, y_pred_dtw, average="weighted")

        results = {"acc": acc_dtw, "f1": f1_dtw_weighted}

        results_dict[dataset_name] = results

    res = pd.DataFrame.from_dict(results_dict).T.reset_index()
    res.columns = ["dataset", "DTW_learned", "DTW_learned F1"]

    res.to_csv("results/dtw_opt_singlevar_baseline.csv")
