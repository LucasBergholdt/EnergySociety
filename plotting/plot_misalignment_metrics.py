# ------------------------------------------------------------------------
# Plots recommendation misalignment metrics.
# Run after analyse.py
# ------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter


def read_csv(folder: Path, filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(folder / filename)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def weighted_strict_follow_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    total = df["total_recommendations"].sum()
    return df["strict_followed"].sum() / total if total > 0 else np.nan


def sum_column(df: pd.DataFrame, column: str) -> float:
    return df[column].sum() if column in df.columns else 0.0


def summarise_condition(folder: Path, label: str) -> dict:
    collisions = read_csv(folder, "recommended_other_only_job_collisions.csv")
    donations = read_csv(folder, "recommended_donations_to_self.csv")
    follow = read_csv(folder, "recommendation_follow_rate.csv")
    decision_follow = read_csv(folder, "recommendation_decision_follow_source.csv")

    if "target_type" in follow.columns:
        follow = follow.copy()
        follow["target_type"] = follow["target_type"].astype(str).str.lower()
        own_follow_rate = weighted_strict_follow_rate(follow[follow["target_type"] == "self"])
    else:
        own_follow_rate = np.nan

    total_decisions = sum_column(decision_follow, "total_decisions")
    other_follow_rate = np.nan
    if total_decisions > 0:
        other_follow_rate = sum_column(decision_follow, "other_followed_decisions") / total_decisions

    return {
        "condition": label,
        "total_other_only_collision_recommendations": sum_column(collisions, "total_other_only_collision_recommendation_events"),
        "total_donate_to_self_recommendations": sum_column(donations, "total_donation_to_self_recommendations"),
        "strict_follow_rate_self": own_follow_rate,
        "decision_follow_rate_other": other_follow_rate,
    }


def make_paired_bar_summary(comparisons: list[dict]) -> pd.DataFrame:
    rows = []
    for comparison in comparisons:
        for folder, label in zip(comparison["dirs"], comparison["labels"]):
            row = summarise_condition(Path(folder), label)
            row["comparison"] = comparison["name"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_recommendation_misalignment_paired_bars_decision_follow(comparisons: list[dict], out_dir: Path) -> None:
    ensure_out(out_dir)
    summary = make_paired_bar_summary(comparisons)

    comparison_labels = [c["name"] for c in comparisons]
    display_labels = {
        "coop_vs_comp_0.5": "Baseline",
        "coop_vs_comp_0.0": "No size penalty",
        "coop_vs_comp_0.5_no_memory": "No memory",
        "coop_vs_comp_0.5_sabotage": "Sabotage",
        "coop_vs_comp_0.5_scarce": "Scarce",
    }
    metric_specs = [
        ("total_other_only_collision_recommendations", "A. Collision recommendations", "Raw total", False),
        ("total_donate_to_self_recommendations", "B. Donate-to-self recommendations", "Raw total", False),
        ("strict_follow_rate_self", "C. Follow rate: own", "Rate", True),
        ("decision_follow_rate_other", "D. Follow rate: other", "Rate", True),
    ]

    x = np.arange(len(comparison_labels))
    width = 0.36
    colors = {"comp": "#b13930", "coop": "#3172b3"}
    fig, axes = plt.subplots(2, 2, figsize=(15.5, max(8.2, 0.45 * len(comparison_labels) + 7.2)))

    for ax, (metric_col, title, ylabel, is_rate) in zip(axes.ravel(), metric_specs):
        pivot = summary.pivot_table(index="comparison", columns="condition", values=metric_col, aggfunc="first").reindex(comparison_labels)
        comp_values = pivot["comp"].to_numpy(dtype=float)
        coop_values = pivot["coop"].to_numpy(dtype=float)

        ax.bar(x - width / 2, comp_values, width, label="comp", color=colors["comp"], alpha=0.88)
        ax.bar(x + width / 2, coop_values, width, label="coop", color=colors["coop"], alpha=0.88)

        finite = np.concatenate([comp_values[np.isfinite(comp_values)], coop_values[np.isfinite(coop_values)]])
        if finite.size:
            ymax = finite.max()
            ax.set_ylim(0, min(1.0, ymax + max(0.05, ymax * 0.12)) if is_rate else ymax + max(1.0, ymax * 0.12))
            annotation_offset = 0.015 if is_rate else max(0.15, ymax * 0.015)
        else:
            annotation_offset = 0.015 if is_rate else 0.15

        for idx, (comp_value, coop_value) in enumerate(zip(comp_values, coop_values)):
            if not np.isfinite(comp_value) or not np.isfinite(coop_value):
                continue
            delta = comp_value - coop_value
            label = f"{delta * 100:+.0f} pp" if is_rate else f"{delta:+.0f}"
            ax.text(idx, max(comp_value, coop_value) + annotation_offset, label, ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([display_labels[name] for name in comparison_labels], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.28)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if is_rate:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Paired competitive/cooperative recommendation metrics", fontsize=13, fontweight="bold", y=0.99)
    fig.subplots_adjust(bottom=0.28, hspace=0.42, wspace=0.24)
    save(fig, out_dir / "recommendation_misalignment_paired_bars_decision_follow.png")


def comparison(root: Path, name: str, experiment_folder: str) -> dict:
    return {
        "name": name,
        "dirs": [
            root / "comp_with_discussion" / experiment_folder / "aggregate",
            root / "coop" / experiment_folder / "aggregate",
        ],
        "labels": ["comp", "coop"],
    }


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    comparisons = [
        comparison(root, "coop_vs_comp_0.5", "exponent_0.5"),
        comparison(root, "coop_vs_comp_0.0", "exponent_0.0"),
        comparison(root, "coop_vs_comp_0.5_no_memory", "exponent_0.5_no_memory"),
        comparison(root, "coop_vs_comp_0.5_sabotage", "exponent_0.5_sabotage"),
        comparison(root, "coop_vs_comp_0.5_scarce", "exponent_0.5_scarce"),
    ]
    plot_recommendation_misalignment_paired_bars_decision_follow(comparisons, root / "misalignment")


if __name__ == "__main__":
    main()

