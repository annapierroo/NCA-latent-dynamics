from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def summarize_conditions(
    run_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    def finite_mean(values: Iterable[float]) -> float:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(array.mean()) if array.size else float("nan")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["condition"])].append(row)
    summaries: list[dict[str, object]] = []
    for condition, rows in grouped.items():
        events = np.asarray([bool(row["event_observed"]) for row in rows])
        first = rows[0]
        divergence_steps = [
            int(row["numerical_divergence_step"])
            for row in rows
            if row.get("numerical_divergence_step", "") != ""
        ]
        summary: dict[str, object] = {
            "condition": condition,
            "label": first["label"],
            "family": first["family"],
            "parameter_name": first["parameter_name"],
            "parameter": first["parameter"],
            "reference_kind": first.get("reference_kind", "mature"),
            "reference_step": first.get("reference_step", ""),
            "n": len(rows),
            "events": int(events.sum()),
            "numerical_divergences": len(divergence_steps),
            "mean_numerical_divergence_step": (
                "" if not divergence_steps else float(np.mean(divergence_steps))
            ),
            "survival_fraction_at_horizon": float((~events).mean()),
            "mean_max_stable_hidden_distance": float(
                np.mean([row["max_stable_hidden_relative_l2"] for row in rows])
            ),
            "mean_hidden_step_rms": float(
                np.mean([row["mean_stable_hidden_step_rms"] for row in rows])
            ),
            "mean_final_target_mse": finite_mean(float(row["final_target_mse"]) for row in rows),
        }
        if all("mean_stable_hidden_reference_relative_l2" in row for row in rows):
            summary["mean_hidden_reference_distance"] = finite_mean(
                float(row["mean_stable_hidden_reference_relative_l2"]) for row in rows
            )
        if all("final_target_region_mse" in row for row in rows):
            summary["mean_final_target_region_mse"] = finite_mean(
                float(row["final_target_region_mse"]) for row in rows
            )
        summaries.append(summary)
    return summaries
