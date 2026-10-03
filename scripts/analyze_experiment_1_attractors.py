from __future__ import annotations

import argparse
from pathlib import Path

from latent_nca.experiment_1.temporal_attractor import (
    TemporalAttractorSettings,
    run_temporal_attractor_analysis,
    select_representative_replicate,
)

CONDITIONS = (
    "baseline",
    "restore_beta_0p1",
    "periodic_reset_150",
    "restore_development_step_16_beta_0p1",
    "restore_development_step_48_beta_0p1",
    "restore_development_step_16_period_150",
    "restore_development_step_48_period_150",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mature-state and developmental-state restoration using paper-style "
            "full-state temporal trajectories in reference and pairwise PCA spaces."
        )
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--replicate", type=int)
    parser.add_argument(
        "--analysis-name",
        help="Optional output folder name below <results-dir>/attractor_analysis/.",
    )
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--dense-until", type=int, default=250)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=40)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--pca-seed", type=int, default=40_000)
    parser.add_argument(
        "--update-probability",
        type=float,
        help="Override the experiment update probability; use 1.0 for a deterministic analysis.",
    )
    parser.add_argument("--device")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    replicate = (
        args.replicate
        if args.replicate is not None
        else select_representative_replicate(results_dir)
    )
    analysis_name = args.analysis_name
    if analysis_name is None:
        analysis_name = f"temporal_full_state_development_restore_t0_{args.horizon}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (results_dir / "attractor_analysis" / analysis_name / f"replicate_{replicate:03d}")
    )
    settings = TemporalAttractorSettings(
        horizon=args.horizon,
        dense_until=args.dense_until,
        record_every=args.record_every,
        crop_size=args.crop_size,
        pca_components=args.pca_components,
        pca_seed=args.pca_seed,
        update_probability=args.update_probability,
    )
    result = run_temporal_attractor_analysis(
        results_dir,
        output_dir,
        replicate,
        CONDITIONS,
        settings,
        device_name=args.device,
    )
    print(f"Temporal attractor analysis completed. Results: {result}")


if __name__ == "__main__":
    main()
