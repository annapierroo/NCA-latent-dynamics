# Experiment 2 Results

Research question: does persistence-oriented training produce a more stable and robust latent dynamical regime around the mature morphology than a Growing NCA with continuous mature hidden-state restoration?

This is a compact viewing bundle for gecko and butterfly. Population figures and summary tables use the 30 recorded stochastic rollout replicas per model. Only the three illustrative replicas already selected in each model's local `analysis` folder are published. These are rollout replicas of one checkpoint per model and morphology, not independent trainings. No simulations or analyses were rerun for this export.

## Start Here

| Morphology | Morphological stability | Hidden deviation from control | Anatomical erasure masks |
| --- | --- | --- | --- |
| Gecko | [MSE and IoU](gecko/morphological_stability_recovery.png) | [Hidden RMS](gecko/hidden_divergence_from_control.png) | [Masks](gecko/primitive_erase_masks.png) |
| Butterfly | [MSE and IoU](butterfly/morphological_stability_recovery.png) | [Hidden RMS](butterfly/hidden_divergence_from_control.png) | [Masks](butterfly/primitive_erase_masks.png) |

Each selected replica folder contains morphology snapshots, hidden PCA in 2D and 3D, and lightweight metadata:

| Morphology | Persistent NCA | Growing NCA + restore |
| --- | --- | --- |
| Gecko | [003](gecko/persistent/replicate_003/), [025](gecko/persistent/replicate_025/), [029](gecko/persistent/replicate_029/) | [017](gecko/growing_restore/replicate_017/), [013](gecko/growing_restore/replicate_013/), [001](gecko/growing_restore/replicate_001/) |
| Butterfly | [026](butterfly/persistent/replicate_026/), [010](butterfly/persistent/replicate_010/), [011](butterfly/persistent/replicate_011/) | [023](butterfly/growing_restore/replicate_023/), [006](butterfly/growing_restore/replicate_006/), [018](butterfly/growing_restore/replicate_018/) |

Selection follows the saved `figure_replicas.json`: the three finite control trajectories closest to median time-averaged log1p MSE and mean IoU, with IQR scaling. The same replica is shown for all its perturbations. Selection is illustrative, not a population-level test or a separate best-case selection for each condition.

## Experimental Protocol

- Both models use 16 channels: four visible RGBA and 12 hidden channels, on a 72 x 72 substrate.
- Persistent NCA evolves under its learned update rule. Growing NCA restores hidden channels 10% toward the original mature reference after every update, over the entire substrate including background; RGBA is not directly restored.
- Time zero is target attainment. Perturbations occur once at step 500, and trajectories continue through step 10,000 with stochastic cell-update probability 0.5. Conditions within each replica share the continuation update masks.
- Conditions: unperturbed control, hidden Gaussian noise at 10% and 25%, and four anatomical erasures. Noise acts on hidden channels inside the target alpha mask; amplitudes use the pre-pulse per-channel RMS. Erasure zeros all 16 channels in the selected region.
- Gecko erasures affect leg, head, tail and torso, removing 50 target pixels each (about 9.84%); the selected leg region is completely removed. Butterfly erasures affect upper-left, upper-right and lower-left wing regions, plus the central body, removing 60 target pixels each (about 9.97%). The obsolete lower-right-wing condition is not published.

`provenance.json` records settings, runtime versions, original implementation and input hashes, source protocol, actual erasure sizes, and published conditions. `calibration.json` contains the original development/attainment thresholds. These are distinct from the post-training evaluation thresholds in `analysis_settings.json`.

## Reading the Results

**Morphological stability.** Global raw RGBA MSE and alpha-mask IoU are measured against the target image, including background. Lines show medians across finite replicas; shaded regions are the 25th-75th percentiles (IQR), not confidence intervals. The vertical dashed line marks step 500. Missing or numerically divergent observations are not imputed.

**Hidden deviation.** RMS distance uses all hidden-state values and compares a perturbed trajectory with its own paired unperturbed control at the same step. Zero means agreement with that control, not necessarily recovery of the target morphology. Raw hidden distances can have different scales across trained models.

**PCA.** The basis is fitted separately per model and replica using only hidden reference states before the pulse. Steps 250-374 fit the basis and channel RMS scales; steps 375-499 provide held-out residual diagnostics. All conditions of that model/replica share the same basis; axes are not shared across models or replicas. Axis percentages report explained variance. A compact projection or apparent loop does not establish a mathematical attractor or a basin boundary.

**Snapshots.** These are actual RGBA frames from the selected replica, not average or median images. The column labelled `t=500, pre` precedes the intervention; `t=500` is immediately after it. Snapshot filenames are normalized in this export; image contents are unchanged.

**Recovery statistics.** Post-pulse morphology evaluation requires MSE <= 0.001, alpha IoU >= 0.95, and area ratio between 0.9 and 1.1. After an exit, sustained recovery requires 20 consecutive target states and at least 95% of the final 200 steps inside the target regime. Never-exited trajectories are reported separately. The CSV column `median_recovery_delay_after_exit` measures delay from the pulse among trajectories that exited and recovered, not delay from the first exit. Unrecovered trajectories are not assigned zero recovery time.

Paired hidden return requires RMS distance <= 10% of the initial pulse distance for 20 consecutive steps and at least 95% of the final 200 steps below that threshold. It is undefined for zero initial distance, including the unperturbed control. This tests return toward the paired trajectory, not proof of return to a mathematical attractor.

## Supporting Files

- `recovery_summary.csv`: endpoint-specific eligible counts, successes, numerical failures, recovery fractions with 95% Wilson intervals, and conditional recovery delays. Empty or `nan` values can indicate no eligible trajectories.
- `paired_model_comparison.csv`: Growing-minus-Persistent success fractions on matched eligible rollout IDs, 5,000 paired bootstrap samples for 95% intervals, exact McNemar p-values, and Holm correction within each morphology/checkpoint pair. Morphologies are not pooled.
- `analysis_settings.json`, `figure_replicas.json`, `exclusions.json`, `calibration.json`, `provenance.json`: evaluation definitions, illustrative selection, exclusions, development calibration, and run provenance.
- Selected `metadata.json`: seeds, maturity step, hidden PCA diagnostics, perturbation details, and validation information for the butterfly erasure update. Local machine paths are omitted.

Raw `.npz` trajectories, checkpoints, full per-replica recovery tables, and the other 27 replica folders per model remain local. This bundle is for viewing and interpretation; it is not a complete raw-data deposit and cannot regenerate all figures by itself. Uncommitted code changes and the proposed projected-flow/recovery-map analysis are not part of this publication.
