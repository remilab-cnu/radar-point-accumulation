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
python3 src/foldwise_stats.py    # point-domain ladder (fold-wise pipeline)
python3 src/foldwise_stats2.py   # map factorial + sign-histogram ladder
```

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
