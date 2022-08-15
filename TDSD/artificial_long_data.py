# This script generates sample and length-duplicated "Crop" and "StarLightCurves" datasets
# for performance testing with the Tempo library.
# To replicate Tempo baseline results, get Tempo from https://github.com/MonashTS/tempo,
# compile the nn1dist component, then run DTW 1-NN with the appropriate parameters,
# e.g. ./nn1dist -dist cdtw 0.1 lb-keogh2 -ucr synthetic Crop_X5

import pandas as pd
import os
from utils.data_io import (
    load_from_tsfile,
    write_dataframe_to_tsfile,
)

ROOT = "data/Univariate_ts/"
OUTPUT = "synthetic"

X_train_0, y_train_0 = load_from_tsfile(
    os.path.join(ROOT, "Crop", "Crop_TRAIN.ts")
)

X_test_0, y_test_0 = load_from_tsfile(
    os.path.join(ROOT, "Crop", "Crop_TEST.ts")
)


X_train_1, y_train_1 = load_from_tsfile(
    os.path.join(ROOT, "StarLightCurves", "StarLightCurves_TRAIN.ts")
)

X_test_1, y_test_1 = load_from_tsfile(
    os.path.join(ROOT, "StarLightCurves", "StarLightCurves_TEST.ts")
)


def multiply_len(X, n=2):
    rows = []
    for row_id in range(X.shape[0]):
        series = X.loc[row_id, "dim_0"]
        rows += [pd.concat([series] * n).reset_index(drop=True)]
    return pd.DataFrame(pd.Series(rows, name="dim_0"), dtype="object")


def multiply_samples(X, n=2):
    res = pd.concat([X] * n, axis=0, ignore_index=True)
    return res


if __name__ == "__main__":
    for mul in range(1, 9):
        Xm_train = multiply_samples(X_train_0, mul)
        Xm_test = multiply_samples(X_test_0, mul)

        write_dataframe_to_tsfile(
            Xm_train, OUTPUT, f"Crop_X{mul}", class_value_list=y_train_0, fold="_TRAIN"
        )
        write_dataframe_to_tsfile(
            Xm_test, OUTPUT, f"Crop_X{mul}", class_value_list=y_test_0, fold="_TEST"
        )

    for mul in range(1, 9):
        Xm_train = multiply_len(X_train_1, mul)
        Xm_test = multiply_len(X_test_1, mul)

        write_dataframe_to_tsfile(
            Xm_train, OUTPUT, f"StarLightCurves_X{mul}", class_value_list=y_train_1, fold="_TRAIN"
        )
        write_dataframe_to_tsfile(
            Xm_test, OUTPUT, f"StarLightCurves_X{mul}", class_value_list=y_test_1, fold="_TEST"
        )
