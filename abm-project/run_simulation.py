"""Batch runner for all experimental conditions."""

import argparse
import csv
import json
from pathlib import Path

from environment import NETWORK_CONDITIONS
from simulation import UPDATE_RULES, run

DEFAULT_N = 100
DEFAULT_T = 1000
DEFAULT_ETA = 0.1
DEFAULT_DELTA = 0.1
DEFAULT_SEEDS = 20
DEFAULT_MEASURE_EVERY = 20


def all_runs(conditions, update_rules, seeds, **run_kwargs):
    """Yield run configurations."""
    for condition in conditions:
        for update_rule in update_rules:
            for seed in seeds:
                yield condition, update_rule, seed


def flatten_trajectory(result):
    """Convert trajectory records into CSV rows."""
    rows = []
    for record in result["trajectory"]:
        rows.append(
            {
                "condition": result["condition"],
                "update_rule": result["update_rule"],
                "seed": result["seed"],
                "centralization": result["centralization"],
                "timestep": record["timestep"],
                "pairwise_consistency": record["pairwise_consistency"],
                "overall_consistency": record["overall_consistency"],
                "prototype_diversity": record["prototype_diversity"],
            }
        )
    return rows


def summary_record(result):
    """Extract end-of-run summary for JSON output."""
    return {
        "condition": result["condition"],
        "update_rule": result["update_rule"],
        "seed": result["seed"],
        "centralization": result["centralization"],
        "hub_id": result["hub_id"],
        "agent_degrees": result["agent_degrees"],
        "hub_initial_prototypes": result["hub_initial_prototypes"],
        "hub_final_prototypes": result["hub_final_prototypes"],
        "initial_prototypes": result["initial_prototypes"],
        "final_prototypes": result["final_prototypes"],
        "final_pairwise_consistency": result["final_pairwise_consistency"],
        "final_overall_consistency": result["final_overall_consistency"],
        "final_prototype_diversity": result["final_prototype_diversity"],
        "convergence_timestep_pairwise": result["convergence_timestep_pairwise"],
        "convergence_timestep_overall": result["convergence_timestep_overall"],
    }


def write_results(output_dir, trajectory_rows, summaries):
    """Write CSV trajectories and JSON summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "trajectories.csv"
    summary_path = output_dir / "summaries.json"

    fieldnames = [
        "condition",
        "update_rule",
        "seed",
        "centralization",
        "timestep",
        "pairwise_consistency",
        "overall_consistency",
        "prototype_diversity",
    ]
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    return trajectory_path, summary_path


def run_all(
    output_dir="results",
    conditions=None,
    update_rules=None,
    seeds=None,
    n=DEFAULT_N,
    T=DEFAULT_T,
    eta=DEFAULT_ETA,
    delta=DEFAULT_DELTA,
    measure_every=DEFAULT_MEASURE_EVERY,
):
    """Run the full experimental sweep and save outputs."""
    conditions = list(conditions or NETWORK_CONDITIONS)
    update_rules = list(update_rules or UPDATE_RULES)
    seeds = list(seeds if seeds is not None else range(DEFAULT_SEEDS))

    trajectory_rows = []
    summaries = []

    total = len(conditions) * len(update_rules) * len(seeds)
    completed = 0

    for condition, update_rule, seed in all_runs(conditions, update_rules, seeds):
        result = run(
            condition=condition,
            update_rule=update_rule,
            seed=seed,
            n=n,
            T=T,
            eta=eta,
            delta=delta,
            measure_every=measure_every,
        )
        trajectory_rows.extend(flatten_trajectory(result))
        summaries.append(summary_record(result))
        completed += 1
        print(
            f"[{completed}/{total}] condition={condition} "
            f"rule={update_rule} seed={seed} "
            f"C={result['centralization']:.3f}"
        )

    return write_results(output_dir, trajectory_rows, summaries)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ABM emergent label simulations.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of agents.")
    parser.add_argument("--T", type=int, default=DEFAULT_T, help="Number of timesteps.")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA, help="Learning rate.")
    parser.add_argument(
        "--delta", type=float, default=DEFAULT_DELTA, help="Ambiguity threshold."
    )
    parser.add_argument(
        "--measure-every",
        type=int,
        default=DEFAULT_MEASURE_EVERY,
        help="Measure consistency every N timesteps.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run with fewer seeds and timesteps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = range(3 if args.quick else DEFAULT_SEEDS)
    T = 200 if args.quick else args.T
    trajectory_path, summary_path = run_all(
        output_dir=args.output_dir,
        seeds=seeds,
        n=args.n,
        T=T,
        eta=args.eta,
        delta=args.delta,
        measure_every=args.measure_every,
    )
    print(f"Wrote {trajectory_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
