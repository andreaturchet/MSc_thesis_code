# MSc Thesis MaxEnt Module Experiments

This repository contains a self-contained notebook for running calibrated
MaxEnt module experiments on MNIST.

## Scope

The experiment trains sparse class-specialized neural modules and compares them
across three complementary levels:

1. **Structural level**: sparse masks and weights.
2. **Representational level**: activations, CKA, Procrustes distance, activation
   correlations, and participation ratio.
3. **Functional level**: output agreement, rewarded accuracy, non-rewarded
   entropy, and weight-sum composition.

The main question is whether modules trained for the same rewarded class are
more similar than modules trained for different classes after accounting for
hidden-neuron permutation symmetries.

## Main File

- `run_calibrated_module_experiments_colab.ipynb`

The notebook contains the full pipeline: model definition, data loading,
training, pruning, metric computation, controls, aggregation, plotting, and
report generation.

## Recommended Usage

Open the notebook in Colab, Kaggle, or another GPU notebook environment.

The notebook detects the runtime automatically:

- Colab: outputs are written under `MyDrive/tesi/` after mounting Google Drive.
- Kaggle: outputs are written under `/kaggle/working/tesi/`.
- Local execution: outputs are written under `raw/experiments/`.

Start with the smoke test:

```python
SMOKE = True
FULL_10X10 = False
```

The smoke test runs a small 2-class x 2-seed configuration and checks the
calibration controls.

For the full 10x10 run:

```python
SMOKE = False
FULL_10X10 = True
```

This configuration trains:

- 10 MNIST classes;
- 10 seeds;
- 100 sparse MaxEnt modules;
- 15 IMP pruning rounds;
- 20 percent pruning per round;
- 10 epochs per pruning round;
- a fixed shared evaluation set.

## Pipeline

The notebook executes these stages:

1. Check the runtime and select GPU or CPU.
2. Mount persistent storage and define the output directory.
3. Load MNIST and build a fixed evaluation subset.
4. Train one sparse MaxEnt module for each `(class, seed)` pair.
5. Save checkpoints, masks, configs, metrics, and activations.
6. Build control and experimental pair categories:
   - identical control;
   - permuted control;
   - random density-matched baseline;
   - same-class different-seed pairs;
   - different-class same-seed pairs;
   - different-class different-seed pairs.
7. Compute structural, representational, and functional metrics.
8. Run aggregate summaries and significance tests.
9. Generate figures, CSV/JSON outputs, and Markdown reports.

## Outputs

For the full run, the default output folder is:

```text
Colab:  MyDrive/tesi/calibrated_module_metrics_full_10x10/
Kaggle: /kaggle/working/tesi/calibrated_module_metrics_full_10x10/
```

Important outputs:

- `report_concise.md`: short summary of the main results.
- `report_full.md`: complete methodological report and tables.
- `pairwise_summary.csv`: aggregated metrics by pair category and layer.
- `significance_tests.csv`: same-class versus cross-class tests.
- `pairwise_results.json`: full pair-level metric data.
- `figures/`: generated plots.
- `checkpoints/`, `masks/`, `configs/`, `metrics/`, `activations/`: reproducibility artifacts.

## Resume Behavior

The run is resume-safe. If the runtime disconnects, rerun the notebook with the
same configuration. Completed modules are skipped automatically unless:

```python
args.overwrite = True
```

is enabled.

To recompute metrics and reports from saved checkpoints without retraining,
enable:

```python
args.no_train = True
```

## Notes

- `args.barrier = False` by default for the full 10x10 run because interpolation
  barriers are optional and expensive on 100 modules.
- The notebook intentionally uses MNIST only and fails if MNIST cannot be
  loaded.
- On Kaggle, Google Drive mounting is not used; artifacts are saved to
  `/kaggle/working/tesi/` and should be downloaded or committed as Kaggle output
  when the run finishes.
- A multi-GPU runtime is not automatically used by the current configuration;
  splitting seeds across separate runs is the simplest way to parallelize.
