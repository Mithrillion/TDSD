This repository reproduces the experiment results in the paper Time-Delay Path Signatures as Features Corresponding to Constrained Dynamic
Time Warping.

1. Environment setup

Here are the key conda/pypi packages used in the experiments:

* autorank                  1.1.2     (manually patched to fix CD digrams - see https://github.com/sherbold/autorank/issues/10)
* cudatoolkit               11.3.1
* cuml*                      22.02.00      (part of rapids)
* dtaidistance              2.3.6
* holoviews*                 1.14.6        (for plot generation only)
* pykeops                   2.1       (additional requirements on https://www.kernel-operations.io/keops/python/installation.html)
* optuna                    2.10.1
* rapids*                    22.02.00      (see https://rapids.ai/start.html#get-rapids. the cudatoolkit version can be forced to agree with PyTorch, e.g., "cudatoolkit=11.3")
* signatory                 1.2.6

(\* Optional. cuml is only needed for GPU implementation of PCA. sklearn's PCA will be used as fallback.)

Other software package(s) for benchmarking:

* Tempo*     https://github.com/MonashTS/tempo@7b12bb8       (see the original repository for compilation guide)

2. Data setup

There should be three folders alongside the scripts: `data/`, `results/` and `synthetic/`.

To replicate our experiment results, the UCR Archive and the SITS dataset should be placed in the `data/` folder.

The UCR Archive (univariate) can be downloaded at http://www.timeseriesclassification.com/dataset.php. The `.arff` version is used for most experiments, but to also run baseline experiments with Tempo, the `.ts` files will also be needed. The files should be placed in `data/Univariate_arff` (or `data/Univariate_ts`).

The SITS dataset can be downloaded at http://bit.ly/SDM201. The files should be placed in `data/SITS_2006_NDVI_C/` (with subfolders of individual validation folds).

3. Scripts

`artificial_long_data.py` - generates `.ts` format files of sample and length-duplicated datasets for run time testing with `Tempo`.

`dtw_optimised_singlevar_baseline.py` - evaluates Euclidean distance and `dtaidistance` DTW performance baselines.

`performance_comparison.py` - Run time comparison of `dtaidistance` against `TDPCA-SigDist`, with pre-computed `Tempo` times.

`sits_timing.py` - evaluates the run time of `TDPCA-SigDist` on the SITS dataset.

`td_sig_singlevar_cv.py` - evaluates `TDPCA-SigDist` performance on the UCR Archive.
