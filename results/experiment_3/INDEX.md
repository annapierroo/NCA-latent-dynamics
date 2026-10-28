# Experiment 3 Results

Can spatially transplanted hidden states guide the development of a different
anatomical structure in a mature NCA?

This bundle contains results from the original [Experiment 3 script](../../scripts/experiment_3.py)
for gecko and planaria, using Persistent NCA and Growing NCA with hidden-state
restoration. Each condition has 30 stochastic rollout replicas from one trained
checkpoint per model and morphology. These are rollout replicas, not independent
trainings. The multitarget experiment is separate and is not included here.

## Browse the Results

| Manipulation | Morphology | Population Curves | Snapshots | Summary CSV | Horizon |
| --- | --- | --- | --- | --- | --- |
| Transplant | Gecko | [Metrics](transplant/gecko/analysis/morphology_metrics.png) | [Persistent](transplant/gecko/analysis/persistent/) / [Growing](transplant/gecko/analysis/growing_restore/) | [CSV](transplant/gecko/analysis/per_run_summary.csv) | 6,000 |
| Transplant | Planaria | [Metrics](transplant/planaria/analysis/morphology_metrics.png) | [Persistent](transplant/planaria/analysis/persistent/) / [Growing](transplant/planaria/analysis/growing_restore/) | [CSV](transplant/planaria/analysis/per_run_summary.csv) | 6,000 |
| Insertion | Gecko | [Metrics](insertion/gecko/analysis/morphology_metrics.png) | [Persistent](insertion/gecko/analysis/persistent/) / [Growing](insertion/gecko/analysis/growing_restore/) | [CSV](insertion/gecko/analysis/per_run_summary.csv) | 6,000 |
| Insertion | Planaria | [Metrics](insertion/planaria/analysis/morphology_metrics.png) | [Persistent](insertion/planaria/analysis/persistent/) / [Growing](insertion/planaria/analysis/growing_restore/) | [CSV](insertion/planaria/analysis/per_run_summary.csv) | 6,000 |
| Persistent insertion with local restore | Gecko | [Metrics](insertion_persistent_local_restore/gecko/analysis/morphology_metrics.png) | [Persistent](insertion_persistent_local_restore/gecko/analysis/persistent/) | [CSV](insertion_persistent_local_restore/gecko/analysis/per_run_summary.csv) | 2,000 |
| Persistent insertion with local restore | Planaria | [Metrics](insertion_persistent_local_restore/planaria/analysis/morphology_metrics.png) | [Persistent](insertion_persistent_local_restore/planaria/analysis/persistent/) | [CSV](insertion_persistent_local_restore/planaria/analysis/per_run_summary.csv) | 2,000 |

Each dataset also contains an anatomical preview, its original `manifest.json`
and `calibration.json`. The manifest records the precise conditions, geometry,
checkpoint hashes, seeds and numerical environment.

## Protocol

- Development starts from the model's seed and ends after five consecutive
  states meet the calibrated target-regime criteria. Manipulation time zero
  starts from this mature state, with its hidden reference held fixed.
- Models have four visible RGBA and twelve hidden channels. Conditions within a
  model and replica share the mature state and stochastic continuation seed;
  cells update with probability 0.5.
- Transplant erases all channels in one anatomical region and copies hidden
  vectors from a different region of the same mature state. The donor copy is
  spatially rotated and translated to align with the body. RGBA is not copied
  from the donor. Gecko: leg from tail, head from leg, tail from head. Planaria:
  head from tail and tail from head.
- Insertion copies hidden vectors into a previously empty region attached to
  the central body, without erasing tissue or directly changing RGBA: a second
  tail for gecko and a second head for planaria.
- Transplanted or inserted hidden vectors are frozen through updates 1-300.
  The remaining state evolves according to the model and restoration condition.
- Growing NCA applies beta = 0.1 to all hidden channels over the entire grid
  after every update. During freezing, the graft values override restoration
  inside its footprint. After release, transplant conditions compare the
  original mature reference with a reference modified only inside that
  footprint to retain the donor hidden vectors. Growing insertion uses the
  modified reference after release.
- Persistent NCA normally evolves freely after freezing. The separate local
  restore variant adds beta = 0.1 only inside the inserted footprint, comparing
  it with the same insertion released without restoration and an unperturbed
  control. Its 2,000-step horizon differs from the 6,000-step main experiments.

## Reading the Figures

Population curves show median and interquartile range across finite replicas.
MSE uses raw RGBA; images are clipped for display. The vertical dashed line marks
the end of the 300-step freeze. Numerically invalid branches terminate and their
unavailable measurements are not replaced with zeros.

The four panels compare RGBA MSE and alpha-mask IoU against the original target
and against an analytical composite with the donor anatomy. The composite is an
evaluation image only; it is never supplied as visible state to the NCA.

Each model has three snapshot figures selected by proximity of the control's
time-averaged MSE to the median control value. They show actual individual
replicas, not averaged images. All conditions in a figure use the same replica.

The CSVs include final MSE, IoU, area ratio, target retention, ectopic growth,
numerical-failure step, first sustained target-regime entry, and the fraction of
the final 200 steps within that regime. Sustained entry requires 20 consecutive
states satisfying the calibrated original-target criteria. An empty final metric
indicates an unavailable numerically valid endpoint, not a zero error.

## Reproduce the Runs

Use the existing Growing and Persistent checkpoints for both morphologies and
install the project's `analysis` dependencies. These commands reproduce the
saved run settings from the project root:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.experiment_3 run \
  --manipulation transplant --morphologies gecko planaria \
  --replicates 30 --horizon 6000 --freeze-steps 300 \
  --output-root reports/experiment_3/transplant

PYTHONPATH=src:. .venv/bin/python -m scripts.experiment_3 run \
  --manipulation insertion --morphologies gecko planaria \
  --replicates 30 --horizon 6000 --freeze-steps 300 \
  --output-root reports/experiment_3/insertion

PYTHONPATH=src:. .venv/bin/python -m scripts.experiment_3 run \
  --manipulation insertion --morphologies gecko planaria --systems persistent \
  --persistent-local-beta 0.1 --replicates 30 --horizon 2000 --freeze-steps 300 \
  --output-root reports/experiment_3/insertion_persistent_local_restore
```

The publication files are byte-for-byte copies of existing local outputs.
[export_manifest.json](export_manifest.json) records their sizes and SHA-256
hashes. Raw trajectory archives and checkpoints remain outside this bundle;
no training, simulation or analysis was rerun to publish these results.
