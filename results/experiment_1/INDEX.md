# Experiment 1 Results

Research question: can manipulating hidden states delay or prevent the long-term degeneration of a Growing NCA after it reaches the target morphology?

This folder publishes existing figures and lightweight supporting data for gecko, butterfly, spider, and planaria. No simulations were rerun for this export. Files copied from `reports/experiment_1` are unchanged; the original reports and raw archives remain local.

## Start Here

Each morphology has 30 paired stochastic evaluation rollouts (IDs 000-029) for each of nine conditions. All conditions within a replicate start from the same mature state and use the same continuation seed. The four morphologies are not pooled as independent replicates. These are rollout replicates, not 30 independently trained models.

| Morphology | Population morphology metrics | Hidden / visible dynamics | Representative snapshots | Full-state PCA |
| --- | --- | --- | --- | --- |
| Gecko | [Metrics](gecko/core_morphology_metrics.png) | [Hidden](gecko/hidden_channel_dynamics.png) / [RGBA](gecko/visible_channel_dynamics.png) | [Replica 021](gecko/representative_replica_021.png) | [Replica 005](gecko/attractor_analysis/replicate_005/) |
| Butterfly | [Metrics](butterfly/core_morphology_metrics.png) | [Hidden](butterfly/hidden_channel_dynamics.png) / [RGBA](butterfly/visible_channel_dynamics.png) | [Replica 028](butterfly/representative_replica_028.png) | [Replica 012](butterfly/attractor_analysis/replicate_012/) |
| Spider | [Metrics](spider/core_morphology_metrics.png) | [Hidden](spider/hidden_channel_dynamics.png) / [RGBA](spider/visible_channel_dynamics.png) | [Replica 025](spider/representative_replica_025.png) | [Replica 004](spider/attractor_analysis/replicate_004/) |
| Planaria | [Metrics](planaria/core_morphology_metrics.png) | [Hidden](planaria/hidden_channel_dynamics.png) / [RGBA](planaria/visible_channel_dynamics.png) | [Replica 025](planaria/representative_replica_025.png) | [Replica 002](planaria/attractor_analysis/replicate_002/) |

## Conditions

All interventions act only on hidden channels 4-15 over the entire substrate, including the background. Visible RGBA channels 0-3 always follow the learned NCA rule.

| File identifier | Intervention after target attainment |
| --- | --- |
| `baseline` | Unconstrained control. |
| `hidden_freeze` | Replace all 12 hidden channels with their mature values after every update. |
| `hidden_freeze_first_6` | Replace the first six hidden channels with their mature values after every update. |
| `restore_beta_0p1` | After every update, move hidden values 10% toward their mature reference. |
| `periodic_reset_150` | Reset hidden values to the mature reference every 150 steps; otherwise allow free evolution. |
| `restore_development_step_16_beta_0p1` | Continuous 10% restoration toward the same rollout's development-step-16 reference. |
| `restore_development_step_48_beta_0p1` | Continuous 10% restoration toward the same rollout's development-step-48 reference. |
| `restore_development_step_16_period_150` | Reset hidden values to the development-step-16 reference every 150 steps. |
| `restore_development_step_48_period_150` | Reset hidden values to the development-step-48 reference every 150 steps. |

## Protocol

- One single-target Growing NCA checkpoint per morphology, with four RGBA and 12 hidden channels.
- Calibration: 32 unperturbed post-training development rollouts, sampled at steps 64-96, using acceptance quantile 0.95 and alpha threshold 0.1. These are not training batches.
- Maturity: five consecutive states inside the calibrated MSE, IoU, and area-ratio bounds. Time zero in the plots is target attainment.
- Evaluation: 10,000 steps after attainment, with stochastic cell-update probability 0.5.
- Sustained morphology failure: the first of ten consecutive states outside the calibrated bounds. No observed sustained failure by the horizon is recorded as right-censored.
- Hidden and visible channel measurements are sampled every ten steps.
- Substrate: 72 x 72 for gecko, butterfly, and planaria; 72 x 67 for the aspect-preserving spider target.

Exact configuration, thresholds, seeds, input hashes, exclusions, and original runtime versions are retained in each morphology's CSV/JSON files. Local absolute paths in metadata describe the original run machine; checkpoint and target paths in `resolved_config.json` are relative to the repository.

## Reading the Figures

**Core morphology metrics.** Solid lines are medians of finite measurements across the paired replicates. Dark bands show the 25th-75th percentiles (IQR); light bands show the 5th-95th percentiles. These bands describe spread, not confidence intervals. Legend counts show finite values at the first and final steps. Divergent trajectories do not contribute after measurements become non-finite. Gray dashed lines mark calibrated global MSE and IoU bounds, not the target image's own error. Target-region MSE measures errors only where the target alpha mask is present; it has no separate calibrated bound.

**Hidden and visible dynamics.** Heatmaps show per-channel median deviation from the mature reference (relative L2) and median consecutive-step activity (RMS). They include the complete substrate, including background. Color scales are logarithmic and clipped at the displayed limits; saturation does not mean equal values. Gray cells indicate unavailable finite measurements, not zero activity.

**Representative snapshots.** Each image shows one actual paired rollout across all conditions, not a pixelwise average or median image. Selection minimizes an IQR-scaled distance to population metric medians across conditions and time, using target-region MSE, IoU, and area ratio; missing measurements are penalized. Scores and selected ID are in `representative_selection.csv`. The PCA diagnostic replica is selected separately and is not necessarily this snapshot replica.

**Final population galleries.** Each `final_t10000__<condition>.png` shows all 30 rollout IDs for that condition at step 10,000. `Numerically diverged` means no valid saved metric state at that time. `Replay not verified` instead means the replay did not reproduce saved measurements within tolerance. The spider control has one such case, replica 016; its [replay diagnostics](spider/final_t10000__baseline.replay_mismatches.csv) are included. Do not count that panel as a demonstrated numerical divergence.

**Development and original snapshots.** `development_to_target.png` shows a single saved development trajectory. Butterfly additionally retains `morphology_snapshots.png`, the original single-rollout montage; it is not the median-selected montage.

**Full-state temporal PCA.** Each PCA folder contains reference-space and pairwise 2D trajectories, pairwise 3D trajectories, coordinates, and metadata for one diagnostic rollout. All nine conditions are included. Representation uses all 16 channels over the full substrate, without spatial pooling. Sampling is every step through 250 and every ten steps afterwards, including the final step when numerically available. Numerical divergence can truncate a trajectory before 10,000.

The shared reference PCA is fitted on the mature-state ensemble and the selected control's states inside the target regime. Each pairwise PCA is fitted on the control and one condition; its axes are not directly comparable with those in other panels. Unequal temporal sampling also affects the fit. A compact trajectory or apparent loop does not prove a mathematical attractor. These single-rollout projections are exploratory diagnostics, not population-level tests.

## Supporting Files

Within each morphology:

- `conditions.csv`: condition labels, intervention parameters, and reference definitions.
- `per_run_summary.csv`: 270 original condition-by-rollout records, including final metrics, sustained-failure information, numerical divergence, development steps, and seeds. Final values may be non-finite after divergence.
- `representative_selection.csv`: scores for all 30 snapshot candidates and selected ID.
- `resolved_config.json`, `target_regime.json`, `metadata.json`, and `excluded_replicates.json`: configuration and provenance.
- `attractor_analysis/replicate_NNN/`: three PCA figures, reference and pairwise coordinate CSVs, and analysis metadata. Pairwise coordinates from different conditions use different bases.
- `*.replay_mismatches.csv`, when present: unresolved snapshot replay discrepancies.

Raw per-step population metrics, channel arrays, saved states, full-state PCA features, and fitted PCA model arrays (`.npz`/`.pt`) are intentionally not uploaded. This is a viewing and interpretation bundle, not a complete raw-data deposit: the population plots and PCA fits cannot be regenerated from this folder alone. No Experiment 2 outputs are included.
