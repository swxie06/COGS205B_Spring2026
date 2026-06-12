"""Hypothesis evaluation and figure generation from simulation outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import population_prototypes, prototype_set_distance

UPDATE_RULES = ("unconditional", "conflict", "ambiguous")
H4_REPRESENTATIVE_CONDITION = "m2"
CONVERGENCE_THRESHOLD = 0.7


def load_trajectories(path):
    """Load trajectory CSV into structured records."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "condition": row["condition"],
                    "update_rule": row["update_rule"],
                    "seed": int(row["seed"]),
                    "centralization": float(row["centralization"]),
                    "timestep": int(row["timestep"]),
                    "pairwise_consistency": float(row["pairwise_consistency"]),
                    "overall_consistency": float(row["overall_consistency"]),
                    "prototype_diversity": float(row["prototype_diversity"]),
                }
            )
    return rows


def load_summaries(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def group_by(keys, records):
    grouped = defaultdict(list)
    for record in records:
        key = tuple(record[k] for k in keys)
        grouped[key].append(record)
    return grouped


def mean_trajectory(records, metric_key):
    by_timestep = defaultdict(list)
    for record in records:
        by_timestep[record["timestep"]].append(record[metric_key])
    timesteps = sorted(by_timestep)
    means = [float(np.mean(by_timestep[t])) for t in timesteps]
    return timesteps, means


def condition_order(conditions):
    """Sort conditions: lattice, m1, m2, m5, m10, star."""
    priority = {"lattice": 0, "star": 99}

    def sort_key(c):
        if c in priority:
            return priority[c]
        if c.startswith("m") and c[1:].isdigit():
            return int(c[1:])
        return 50

    return sorted(conditions, key=sort_key)


def plot_h1_consistency_trajectories(trajectories, output_dir, metric_key, filename, title):
    """Plot consistency trajectories with one subplot per update strategy."""
    grouped = group_by(["condition", "update_rule"], trajectories)
    conditions = condition_order({k[0] for k in grouped})

    fig, axes = plt.subplots(1, len(UPDATE_RULES), figsize=(14, 4), sharey=True)
    if len(UPDATE_RULES) == 1:
        axes = [axes]

    for ax, rule in zip(axes, UPDATE_RULES):
        for condition in conditions:
            key = (condition, rule)
            if key not in grouped:
                continue
            records = grouped[key]
            timesteps, means = mean_trajectory(records, metric_key)
            centralization = float(np.mean([r["centralization"] for r in records]))
            ax.plot(
                timesteps,
                means,
                label=f"{condition} (C={centralization:.2f})",
            )
        ax.set_title(rule)
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel(metric_key.replace("_", " ").title())
    fig.suptitle(title)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_h1_convergence_time(summaries, output_dir, metric_key, convergence_key, filename, title):
    """Bar plot of convergence time vs centralization with faded bars for non-converged."""
    grouped = group_by(["condition", "update_rule"], summaries)
    conditions = condition_order({k[0] for k in grouped})
    inferred_max = 0
    for s in summaries:
        for key in ("convergence_timestep_pairwise", "convergence_timestep_overall"):
            val = s.get(key)
            if val is not None:
                inferred_max = max(inferred_max, val)
    plot_max = max(inferred_max, 200)

    fig, axes = plt.subplots(1, len(UPDATE_RULES), figsize=(14, 4), sharey=True)
    if len(UPDATE_RULES) == 1:
        axes = [axes]

    for ax, rule in zip(axes, UPDATE_RULES):
        x_labels = []
        heights = []
        colors = []
        hatches = []
        centralizations = []

        for condition in conditions:
            key = (condition, rule)
            if key not in grouped:
                continue
            group = grouped[key]
            conv_times = [s[convergence_key] for s in group]
            reached = [t for t in conv_times if t is not None]
            mean_c = float(np.mean([s["centralization"] for s in group]))
            centralizations.append(mean_c)
            x_labels.append(condition)

            if reached:
                heights.append(float(np.mean(reached)))
                colors.append("steelblue")
                hatches.append("")
            else:
                heights.append(float(plot_max))
                colors.append("lightgray")
                hatches.append("///")

        x = np.arange(len(x_labels))
        bars = ax.bar(x, heights, color=colors, edgecolor="black")
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            if hatch:
                bar.set_alpha(0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(rule)
        ax.axhline(CONVERGENCE_THRESHOLD, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Timesteps to 70% consistency")
    fig.suptitle(title + " (faded = not reached)")
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compute_h2_distances(summary):
    """Distance from each agent's initial prototypes to final population prototypes."""
    agents_initial = [np.array(p) for p in summary["initial_prototypes"]]
    agents_final = [np.array(p) for p in summary["final_prototypes"]]

    class _Proxy:
        def __init__(self, prototypes):
            self.prototypes = prototypes

    final_pop = population_prototypes([_Proxy(p) for p in agents_final])
    distances = []
    degrees = summary["agent_degrees"]
    for initial, degree in zip(agents_initial, degrees):
        distances.append(
            {
                "degree": degree,
                "distance": prototype_set_distance(initial, final_pop),
            }
        )
    return distances


def plot_h2_distance_vs_degree(summaries, output_dir):
    """Scatter distance from initial prototypes to final population vs agent degree."""
    records = []
    for summary in summaries:
        if summary["update_rule"] != "unconditional":
            continue
        for entry in compute_h2_distances(summary):
            records.append(
                {
                    "degree": entry["degree"],
                    "distance": entry["distance"],
                    "condition": summary["condition"],
                    "centralization": summary["centralization"],
                }
            )

    if not records:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    conditions = condition_order({r["condition"] for r in records})
    for condition in conditions:
        subset = [r for r in records if r["condition"] == condition]
        degrees = [r["degree"] for r in subset]
        distances = [r["distance"] for r in subset]
        mean_c = float(np.mean([r["centralization"] for r in subset]))
        ax.scatter(
            degrees,
            distances,
            alpha=0.4,
            s=15,
            label=f"{condition} (C={mean_c:.2f})",
        )

    ax.set_xlabel("Agent degree")
    ax.set_ylabel("Distance from initial to final population prototypes")
    ax.set_title("H2: Hub bias — prototype distance vs degree")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "h2_distance_vs_degree.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_h3_diversity_trajectories(trajectories, output_dir):
    """Prototype diversity trajectories across conditions, one subplot per strategy."""
    grouped = group_by(["condition", "update_rule"], trajectories)
    conditions = condition_order({k[0] for k in grouped})

    fig, axes = plt.subplots(1, len(UPDATE_RULES), figsize=(14, 4), sharey=True)
    if len(UPDATE_RULES) == 1:
        axes = [axes]

    for ax, rule in zip(axes, UPDATE_RULES):
        for condition in conditions:
            key = (condition, rule)
            if key not in grouped:
                continue
            records = grouped[key]
            timesteps, means = mean_trajectory(records, "prototype_diversity")
            centralization = float(np.mean([r["centralization"] for r in records]))
            ax.plot(
                timesteps,
                means,
                label=f"{condition} (C={centralization:.2f})",
            )
        ax.set_title(rule)
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Prototype diversity (std)")
    fig.suptitle("H3: Prototype diversity trajectories")
    fig.tight_layout()
    path = output_dir / "h3_diversity_trajectories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_h3_distance_matrix(summaries, output_dir):
    """Heatmap of pairwise prototype distances for low vs high centralization."""
    low = next(
        (s for s in summaries if s["condition"] == "lattice" and s["update_rule"] == "unconditional"),
        None,
    )
    high = next(
        (s for s in summaries if s["condition"] == "star" and s["update_rule"] == "unconditional"),
        None,
    )
    if not low or not high:
        return None

    def distance_matrix(summary):
        protos = [np.array(p).reshape(-1) for p in summary["final_prototypes"]]
        n = len(protos)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(protos[i] - protos[j]))
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, summary, title in zip(
        axes,
        [low, high],
        ["Low centralization (lattice)", "High centralization (star)"],
    ):
        matrix = distance_matrix(summary)
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(f"{title}\nseed={summary['seed']}")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Agent")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("H3: Pairwise prototype distance matrix (representative runs)")
    fig.tight_layout()
    path = output_dir / "h3_distance_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_h4_update_rules(summaries, output_dir, condition=H4_REPRESENTATIVE_CONDITION):
    """Four bar plots comparing update strategies at a representative centralization level."""
    group = [s for s in summaries if s["condition"] == condition]
    if not group:
        return None

    by_rule = group_by(["update_rule"], group)
    rules = [r for r in UPDATE_RULES if (r,) in by_rule]
    if not rules:
        return None

    conv_pairwise = []
    conv_reached = []
    final_pairwise = []
    final_overall = []
    final_diversity = []

    inferred_max = 0
    for rule in rules:
        records = by_rule[(rule,)]
        conv = [
            s["convergence_timestep_pairwise"]
            for s in records
            if s["convergence_timestep_pairwise"] is not None
        ]
        if conv:
            inferred_max = max(inferred_max, max(conv))
        conv_reached.append(bool(conv))
        conv_pairwise.append(float(np.mean(conv)) if conv else 0.0)
        final_pairwise.append(float(np.mean([s["final_pairwise_consistency"] for s in records])))
        final_overall.append(float(np.mean([s["final_overall_consistency"] for s in records])))
        final_diversity.append(float(np.mean([s["final_prototype_diversity"] for s in records])))

    plot_max = max(inferred_max, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    conv_colors = []
    conv_hatches = []
    conv_heights = []
    for height, reached in zip(conv_pairwise, conv_reached):
        if reached:
            conv_heights.append(height)
            conv_colors.append("steelblue")
            conv_hatches.append("")
        else:
            conv_heights.append(float(plot_max))
            conv_colors.append("lightgray")
            conv_hatches.append("///")

    conv_bars = axes[0, 0].bar(rules, conv_heights, color=conv_colors, edgecolor="black")
    for bar, hatch in zip(conv_bars, conv_hatches):
        bar.set_hatch(hatch)
        if hatch:
            bar.set_alpha(0.5)
    axes[0, 0].set_title("Convergence time (pairwise 70%)")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    other_metrics = [
        (final_pairwise, "Final pairwise consistency"),
        (final_overall, "Final overall consistency"),
        (final_diversity, "Final prototype diversity"),
    ]
    for ax, (values, label) in zip(axes.flat[1:], other_metrics):
        ax.bar(rules, values, color="steelblue", edgecolor="black")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    mean_c = float(np.mean([s["centralization"] for s in group]))
    fig.suptitle(
        f"H4: Update strategy comparison (condition={condition}, C={mean_c:.2f})"
    )
    fig.tight_layout()
    path = output_dir / "h4_update_rules.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary(summaries):
    """Print textual summary for each hypothesis."""
    print("=== H1: Centralization accelerates convergence ===")
    by_condition = group_by(
        ["condition"],
        [s for s in summaries if s["update_rule"] == "unconditional"],
    )
    ordered = condition_order([k[0] for k in by_condition.keys()])
    for condition in ordered:
        group = by_condition[(condition,)]
        conv_p = [s["convergence_timestep_pairwise"] for s in group if s["convergence_timestep_pairwise"] is not None]
        conv_o = [s["convergence_timestep_overall"] for s in group if s["convergence_timestep_overall"] is not None]
        mean_conv_p = float(np.mean(conv_p)) if conv_p else float("nan")
        mean_conv_o = float(np.mean(conv_o)) if conv_o else float("nan")
        mean_final_p = float(np.mean([s["final_pairwise_consistency"] for s in group]))
        mean_final_o = float(np.mean([s["final_overall_consistency"] for s in group]))
        mean_c = float(np.mean([s["centralization"] for s in group]))
        print(
            f"  condition={condition} C={mean_c:.3f} "
            f"conv_pairwise={mean_conv_p:.1f} conv_overall={mean_conv_o:.1f} "
            f"final_pairwise={mean_final_p:.3f} final_overall={mean_final_o:.3f}"
        )

    print("\n=== H4: Update strategy comparison ===")
    by_rule = group_by(["update_rule"], summaries)
    for rule, group in sorted(by_rule.items()):
        mean_final_p = float(np.mean([s["final_pairwise_consistency"] for s in group]))
        mean_final_o = float(np.mean([s["final_overall_consistency"] for s in group]))
        mean_div = float(np.mean([s["final_prototype_diversity"] for s in group]))
        print(
            f"  rule={rule[0]} final_pairwise={mean_final_p:.3f} "
            f"final_overall={mean_final_o:.3f} diversity={mean_div:.3f}"
        )


def analyze(results_dir="results", figures_dir="figures"):
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(results_dir / "trajectories.csv")
    summaries = load_summaries(results_dir / "summaries.json")

    paths = []
    paths.append(
        plot_h1_consistency_trajectories(
            trajectories,
            figures_dir,
            "pairwise_consistency",
            "h1_pairwise_trajectories.png",
            "H1: Pairwise consistency trajectories",
        )
    )
    paths.append(
        plot_h1_consistency_trajectories(
            trajectories,
            figures_dir,
            "overall_consistency",
            "h1_overall_trajectories.png",
            "H1: Overall consistency trajectories",
        )
    )
    paths.append(
        plot_h1_convergence_time(
            summaries,
            figures_dir,
            "pairwise_consistency",
            "convergence_timestep_pairwise",
            "h1_convergence_pairwise.png",
            "H1: Convergence time (pairwise)",
        )
    )
    paths.append(
        plot_h1_convergence_time(
            summaries,
            figures_dir,
            "overall_consistency",
            "convergence_timestep_overall",
            "h1_convergence_overall.png",
            "H1: Convergence time (overall)",
        )
    )
    path = plot_h2_distance_vs_degree(summaries, figures_dir)
    if path:
        paths.append(path)
    paths.append(plot_h3_diversity_trajectories(trajectories, figures_dir))
    path = plot_h3_distance_matrix(summaries, figures_dir)
    if path:
        paths.append(path)
    path = plot_h4_update_rules(summaries, figures_dir)
    if path:
        paths.append(path)

    print_summary(summaries)
    print("\nWrote figures:")
    for path in paths:
        if path is not None:
            print(f"  {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ABM simulation results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    analyze(args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
