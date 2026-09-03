# Radar Point-Cloud Accumulation — code and measurement records

Code, result records, and per-instance predictions for the manuscript
**"Doppler Velocity in Radar Point-Cloud Recognition: Controlled Attribution
and the Sign-Cancellation Limit of Scalar Accumulation"** (submitted to IEEE
Transactions on Instrumentation and Measurement).

Everything a number in the paper rests on is here: the experiment scripts
(`src/`), the run records with per-arm minimum training accuracies
(`results/*.json`), the saved per-instance predictions that every
subject-clustered interval is computed from (`results/*_preds.npz`), the
statistics scripts that turn predictions into the quoted contrasts
(`src/foldwise_stats*.py`, `figures/paired_stats.py`), and the GPU job
specifications (`jobs/`).

## Reproducing the paper's numbers

The paired contrasts and confidence intervals regenerate from the saved
predictions alone, without datasets or GPUs (numpy only, fixed seeds,
bit-reproducible):

```bash
ln -s results docs                  # the scripts read and write ../docs
python3 src/foldwise_stats.py       # point-domain ladder (fold-wise pipeline)
python3 src/foldwise_stats2.py      # map factorial + sign-histogram ladder
python3 src/variance_components.py  # Sec. III-C variance decomposition
python3 src/cluster_unit_check.py   # Sec. III-C resampling-unit check
python3 src/c_grid_stats.py         # Sec. V-B grid sweep of the coherence index C
```

The statistics scripts are the files that were run for the paper, unedited,
and they resolve their inputs and outputs as `docs/` beside `src/`; the
symbolic link above points that name at `results/`.

`src/variance_components.py` writes `results/variance_components.json`, the
Section III-C decomposition of the primary contrast into subject, seed, and
residual standard deviations (5.3, 0.5, and 2.9 pp) and the 91 / 4 / 5 %
shares of the variance of the mean; its `ep120` entry is the earlier round,
kept for comparison. `src/cluster_unit_check.py` writes
`results/cluster_unit_check.json`, the Section III-C check that resampling
subjects rather than folds leaves the interval essentially unchanged. Those
two are the ep300 scripts; `figures/variance_components.py` is the older
ep120 script behind the earlier figure and is kept as it was.

`src/c_grid_stats.py` reads `results/c_grid_sweep.json` and
`results/c_grid_sweep_preds.npz` and writes `results/c_grid_stats.json`, the
Section V-B sweep of the coherence index C over 16, 32, and 64 map bins;
`src/rep_c_grid_sweep.py` is the training run that produced those two files.
`src/rep_shared_norm_sweep.py` produced `results/shared_norm_sweep.json` and
`results/shared_norm_sweep_preds.npz` for the signed-pair-sum arm under shared
normalization (Table IV, row B2, summarized in
`results/shared_norm_stats.json`), together with the within-dataset C sweep
reported in the supplement's sweep section.
`jobs/vessl_shared_norm_sweep.yaml` is that run's job specification; no job
file was archived for the C-grid sweep.

Retraining from scratch needs the four public datasets; `DATASET_ACQUISITION.md`
records, per dataset, the exact source, file identifiers, byte sizes, and
verified acquisition commands. None of the data is ours to rehost; all four
sets are public releases by their original authors (mHomeGes, MM-Fi, mRI,
BGT60TR13C). The `jobs/*.yaml` files record every training run's budget,
seeds, and entry point; `src/rep_foldwise_ladder.py` is the primary
attribution experiment, `src/rep_hist_ladder.py` and
`src/rep_sign_controls.py` the map-domain cancellation experiments.

## Protocol in one paragraph

Subject-disjoint five-fold evaluation with preprocessing statistics fitted on
training subjects only; capacity-matched arms; information-destroying shuffle
controls at matched input dimensionality; uniform training budgets with a
0.95 minimum-training-accuracy criterion and a 0.90 display-only floor; and
subject-clustered (two-level subject-and-seed bootstrap) intervals on every
claim-bearing contrast. Details: Section III of the paper and the
"Estimators, Interventions, and Model Specifications" section of its
supplement.

## Fold assignment

Folds are a seeded partition of subject identifiers (`kfold(subjects, 5)`
with seed 0, `src/rep_round3.py`), identical across all experiments on a
dataset; per-instance predictions in the `.npz` files carry fold and seed in
their keys.

## Citation

Citation entry will be added on publication. Until then, please cite the
manuscript by title.

## License

MIT (see `LICENSE`). The datasets keep their original licenses.
