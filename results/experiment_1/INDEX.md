# Experiment 1 Results

Can hidden-state interventions preserve a mature Growing NCA morphology over
10,000 further stochastic updates?

This bundle replaces the previous nine-condition results with the corrected
**eight-condition experiment**, including morphology-only hidden freezing and
development-step-16 restoration. Development-step-48 figures and old PCA results
are no longer part of this publication. No simulations were rerun for this export.

## Figures

Each morphology has 30 paired stochastic rollout replicas. A replica uses the
same mature initial state and continuation seed for all eight conditions. These
are not independently trained models; the four morphologies are not pooled.

| Morphology | Population Metrics | Channel Dynamics | Representative Snapshots | Final Population |
| --- | --- | --- | --- | --- |
| Gecko | [MSE, IoU, area](gecko/core_morphology_metrics.png) | [Hidden](gecko/hidden_channel_dynamics.png) / [RGBA](gecko/visible_channel_dynamics.png) | [Replica 014](gecko/morphology_snapshots.png) | [Target-only freeze](gecko/final_t10000__hidden_freeze_morphology.png) |
| Butterfly | [MSE, IoU, area](butterfly/core_morphology_metrics.png) | [Hidden](butterfly/hidden_channel_dynamics.png) / [RGBA](butterfly/visible_channel_dynamics.png) | [Replica 028](butterfly/morphology_snapshots.png) | [Target-only freeze](butterfly/final_t10000__hidden_freeze_morphology.png) |
| Planaria | [MSE, IoU, area](planaria/core_morphology_metrics.png) | [Hidden](planaria/hidden_channel_dynamics.png) / [RGBA](planaria/visible_channel_dynamics.png) | [Replica 025](planaria/morphology_snapshots.png) | [Target-only freeze](planaria/final_t10000__hidden_freeze_morphology.png) |
| Spider | [MSE, IoU, area](spider/core_morphology_metrics.png) | [Hidden](spider/hidden_channel_dynamics.png) / [RGBA](spider/visible_channel_dynamics.png) | [Replica 007](spider/morphology_snapshots.png) | [Target-only freeze](spider/final_t10000__hidden_freeze_morphology.png) |

Each morphology folder also contains `development_to_target.png` and a
`final_t10000__<condition>.png` gallery for every condition, showing all 30 replicas.

## Conditions

Visible RGBA channels always follow the learned NCA rule. Interventions affect
hidden channels only, after each natural update. Except for target-only freezing,
they operate on the complete substrate, including background.

| Identifier | Intervention |
| --- | --- |
| `baseline` | Unconstrained control. |
| `hidden_freeze` | Reset all hidden channels to their mature values after every update. |
| `hidden_freeze_morphology` | Reset all hidden channels only inside the fixed target-alpha mask. Background hidden channels remain free. |
| `hidden_freeze_first_6` | Reset the first six of the twelve hidden channels to their mature values. |
| `restore_beta_0p1` | Move hidden values 10% toward the fixed mature reference after every update. |
| `periodic_reset_150` | Hard reset to the mature hidden reference every 150 updates. |
| `restore_development_step_16_beta_0p1` | Continuous 10% restoration toward this replica's development-step-16 hidden state. |
| `restore_development_step_16_period_150` | Hard reset to the development-step-16 hidden state every 150 updates. |

## Protocol and Interpretation

- One pretrained Growing NCA checkpoint per morphology: four RGBA and twelve hidden channels.
- Target-regime calibration: 32 independent post-training developments, sampled at steps 64-96; acceptance quantile 0.95 and alpha threshold 0.1.
- Maturity: five consecutive states inside the calibrated MSE, IoU and area-ratio bounds. Plot time zero is target attainment.
- Evaluation: 10,000 further updates, cell-update probability 0.5. Sustained failure starts at the first of ten consecutive out-of-regime states.
- Complete NCA states and measurements are checked for numerical validity. After NaN/Inf, the corresponding branch ends permanently; unavailable values are not replaced with zeros.
- Core curves: finite-only median, dark 25th-75th percentile band and light 5th-95th percentile band. These describe rollout variability, not confidence intervals. A curve ends when no valid replicas remain.
- Dashed MSE/IoU lines and the area-ratio band show calibrated target-regime bounds. Target-region MSE has no separately calibrated threshold.
- Channel heatmaps: per-channel median relative-L2 deviation from the mature state and consecutive-step RMS activity, over the whole grid. Color limits are logarithmic and clipped; gray indicates missing data, not inactivity.
- Snapshots: saved frames from one actual paired replica nearest the joint median trajectories, not averaged images and not a new simulation replay. Each row uses the same replica. Snapshot colors are display-clipped; quantitative MSE uses raw RGBA.
- Final galleries: saved states at step 10,000; `Numerically diverged` means that branch no longer has a finite valid state.

Numerically large MPS states use CPU float64 reductions only after a native-dtype
CPU transfer. This preserves the corrected measurement procedure without changing
the learned NCA update rule, intervention strengths or seeds.

## Supporting Data

Each morphology contains:

- `conditions.csv` and `per_run_summary.csv`: eight conditions and 240 condition-by-replica summaries, including seeds, failures and final metrics.
- `resolved_config.json`, `target_regime.json`, `metadata.json` and `excluded_replicates.json`: original configuration, calibrated thresholds and provenance. Absolute paths describe the original run location and are preserved unchanged.
- `representative_selection.csv`: selection scores for all 30 replicas.
- `n_valid_changes.csv`: valid-replica counts at step zero, every count change and the final step. Counts are piecewise constant; forward-fill between these rows to reconstruct the full count series without uploading 80,008 repetitive rows per morphology.

[export_manifest.json](export_manifest.json) lists the source run signatures,
SHA-256 hashes and file sizes for this bundle. PNG files and original CSV/JSON
supporting files are byte-for-byte copies; only `n_valid_changes.csv` is a compact,
lossless representation of the full validity-count CSV.

Raw `.npz`/`.pt` trajectories and intermediate caches remain local. This is a
viewing and interpretation bundle, not a complete raw-data deposit. No Experiment
2, beta-restoration or MPC results are included.

## Updated PCA

The previous PCA figures used a different condition set and are intentionally
removed rather than mixed with these results. PCA remains available in the code.
To generate updated full-state PCA for every current condition, on the local
corrected archives:

```bash
PYTHONPATH=src .venv/bin/python scripts/experiment_1.py attractor \
  --results-root "$HOME/Downloads/NCA_recovery/experiment_1_corrected_2026-10-17" \
  --morphologies gecko butterfly planaria spider \
  --crop-size 0 \
  --record-every 10 \
  --analysis-name temporal_full_state
```

The default analyzes one representative replica per morphology, using all sixteen
state channels and the original device/environment. Outputs are written under
`<morphology>/analysis/attractor/temporal_full_state/replicate_NNN/`. These new PCA
results are not included in the present bundle. PCA geometry alone is not proof
of a mathematical attractor.
