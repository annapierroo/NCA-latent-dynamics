# NCA Latent Dynamics

We investigate how hidden cellular states contribute to morphological stability
and recovery in Neural Cellular Automata. Using trained models, we manipulate
these internal states and observe how visible patterns evolve over time.

## Experiment 1: Controlling Hidden States

**Can we preserve a mature morphology by intervening on its hidden states?**

We study four morphologies: **gecko, butterfly, planaria, and spider**. We compare
eight conditions in Growing NCA:

- Unperturbed control.
- Global hidden-state freezing.
- Hidden-state freezing inside the fixed target-alpha mask only.
- Partial freezing of the first six of twelve hidden channels.
- Continuous restoration toward the mature hidden state, with beta = 0.1.
- Hard reset to the mature hidden state every 150 steps.
- Continuous restoration toward the development-step-16 hidden state, with beta = 0.1.
- Hard reset to the development-step-16 hidden state every 150 steps.

Restoration moves hidden values toward a fixed recorded reference after each
update. Visible RGBA channels remain free to evolve. We examine morphological
metrics, hidden and visible channel activity, snapshots, and full-state PCA.

## Experiment 2: Persistence and Recovery

**Does persistence-oriented training produce more robust dynamics than external
hidden-state restoration?**

We compare **Persistent NCA** with **Growing NCA using continuous mature
hidden-state restoration at beta = 0.1**, for gecko and butterfly.

At step 500 after target attainment, we apply hidden-state Gaussian noise at
10% or 25%, or anatomically localized erasure. Gecko erasures affect one leg,
the head, tail, or torso. Butterfly erasures affect the upper-left, upper-right,
or lower-left wing, or the central body. We also retain an unperturbed control.

We distinguish target-morphology recovery from agreement with the time-matched
unperturbed control. We examine MSE, alpha-mask IoU, hidden-state RMS distances,
hidden-only PCA trajectories, and visual snapshots. Growing restoration acts
over the entire grid, including the background; RGBA is not directly restored.

## Shared Protocol

We follow 30 stochastic rollouts per condition for 10,000 steps after target
attainment. Conditions share the same mature starting state and stochastic
update sequence within each model and replica. These are repeated evaluations
of fixed checkpoints, not independently trained models.

We summarize rollout variability with medians and percentile bands. Numerically
invalid trajectories terminate without zero imputation. PCA projections describe
observed geometry; they do not establish a mathematical attractor.

## Explore the Experiments

- [Experiment 1: figures, protocol, and analysis tables](results/experiment_1/INDEX.md)
- [Experiment 2: figures, protocol, and analysis tables](results/experiment_2/INDEX.md)

Our interactive explorer is in preparation. We will include every recorded
replica, a randomly selected replica on refresh, and synchronized playback of
control and intervention. Additional frames will come from verified replay of
the original simulations, without retraining or image interpolation.

Our NCA baseline follows [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/).
Upstream implementations and their licenses are preserved under `vendor/`.

## Run the Experiments

Use Python 3.10 or newer and install the analysis dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[analysis]"
```

We use RGBA targets at `data/<morphology>.png`, Growing checkpoints at
`checkpoints/growing/<morphology>/model.pt`, and Persistent checkpoints at
`checkpoints/persistent/<morphology>/model.pt`. These commands evaluate existing
models; they do not train them. Device selection is automatic.

```bash
# Experiment 1: all four morphologies, 30 paired rollouts
PYTHONPATH=src .venv/bin/python -m scripts.experiment_1 run --replicates 30
PYTHONPATH=src .venv/bin/python -m scripts.experiment_1 plot
PYTHONPATH=src .venv/bin/python -m scripts.experiment_1 attractor

# Experiment 2: gecko and butterfly, 30 rollouts per model
PYTHONPATH=src .venv/bin/python -m scripts.experiment_2 run --replicates 30
PYTHONPATH=src .venv/bin/python -m scripts.experiment_2 attractor
PYTHONPATH=src .venv/bin/python -m scripts.experiment_2 compare-control
```

We write outputs to `reports/experiment_1` and `reports/experiment_2`. For completed
runs, skip `run` to regenerate analyses from local archives. Experiment 1 PCA
replays the selected trajectories; Experiment 2 control comparisons may replay
missing data. Add `--morphologies` followed by names to process a subset. PCA
figures use representative replicas by default; population statistics use all
included rollouts.
