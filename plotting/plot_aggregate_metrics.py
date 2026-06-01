# ------------------------------------------------------------------------
# Plots graphs.
# Run after analyse.py
# ------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter


DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def _read_csv(folder: Path, filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(folder / filename)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_metric(folders: list[Path], labels: list[str], filename: str) -> pd.DataFrame:
    frames = []
    for folder, label in zip(folders, labels):
        df = _read_csv(folder, filename).copy()
        if df.empty:
            continue
        df.insert(0, "experiment", label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _agent_sort_key(agent_id: str):
    tail = str(agent_id).split("_")[-1]
    return int(tail) if tail.isdigit() else str(agent_id)


def _ordered_agents(df: pd.DataFrame) -> list[str]:
    return sorted(df["agent_id"].dropna().astype(str).unique(), key=_agent_sort_key)


def _agent_model_labels(df: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for agent_id in _ordered_agents(df):
        model_names = df.loc[df["agent_id"] == agent_id, "model_name"].dropna().astype(str).unique()
        labels[agent_id] = f"{agent_id} ({model_names[0]})" if len(model_names) else agent_id
    return labels


def _ordered_values(values: Iterable[str], preferred_order: list[str] | None = None) -> list[str]:
    vals = [v for v in pd.Series(list(values)).dropna().unique()]
    if preferred_order:
        ordered = [v for v in preferred_order if v in vals]
        ordered += sorted([v for v in vals if v not in ordered])
        return ordered
    return sorted(vals)


def _experiment_agent_col(df: pd.DataFrame) -> pd.Series:
    return df["experiment"].astype(str) + " | " + df["agent_id"].astype(str)


def _difficulty_color_map(difficulties: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab10")
    preferred = {"easy": cmap(2), "medium": cmap(0), "hard": cmap(3)}
    return {d: preferred.get(d, cmap(i % 10)) for i, d in enumerate(difficulties)}


def _lighten_color(color: object, amount: float = 0.62) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def plot_energy_over_time_aggregate(folders: list[Path], labels: list[str], out_dir: Path) -> None:
    df = _load_metric(folders, labels, "energy_over_time.csv")
    if df.empty:
        return

    df = df.copy()
    df["std_energy"] = df["std_energy"].fillna(0.0)
    experiments = list(df["experiment"].dropna().unique())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10, max(4, 3.2 * len(experiments))), sharex=True, squeeze=False)

    for ax, exp in zip(axes.flatten(), experiments):
        sub = df[df["experiment"] == exp]
        agent_labels = _agent_model_labels(sub)
        for agent_id in _ordered_agents(sub):
            grp = sub[sub["agent_id"] == agent_id].sort_values("round_num")
            x = grp["round_num"].to_numpy(dtype=float)
            y = grp["avg_energy"].to_numpy(dtype=float)
            n_runs = grp["n_runs"].replace(0, np.nan).to_numpy(dtype=float)
            sem = grp["std_energy"].to_numpy(dtype=float) / np.sqrt(n_runs)
            sem = np.nan_to_num(sem, nan=0.0)
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=agent_labels.get(agent_id, agent_id))
            ax.fill_between(x, y - sem, y + sem, alpha=0.18)

        ax.set_title(f"Mean Agent Energy over Time ({exp})")
        ax.set_ylabel("Energy")
        ax.grid(alpha=0.3)
        if ax is axes.flatten()[0]:
            ax.legend(fontsize=8, loc="upper left", ncol=1, frameon=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes.flatten()[-1].set_xlabel("Round")
    _save(fig, out_dir / "00_energy_over_time_aggregate.png")


def plot_energy_spent(folders: list[Path], labels: list[str], out_dir: Path) -> None:
    df = _load_metric(folders, labels, "energy_spent_per_phase.csv")
    if df.empty:
        return

    phase_cols = ["energy_vote_mean", "energy_decide_mean", "energy_act_mean"]
    rename = {"energy_vote_mean": "Vote", "energy_decide_mean": "Decide", "energy_act_mean": "Act"}
    phase_labels = [rename.get(c, c) for c in phase_cols]
    phase_colors = {"Vote": plt.get_cmap("tab10")(0), "Decide": plt.get_cmap("tab10")(1), "Act": plt.get_cmap("tab10")(2)}

    experiments = list(df["experiment"].dropna().unique())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(9.5, max(4.0, 3.2 * len(experiments))), sharey=True, squeeze=False)
    y_max = float(df[phase_cols].fillna(0).sum(axis=1).max()) * 1.12

    for ax, exp in zip(axes.flatten(), experiments):
        sub = df[df["experiment"] == exp].copy()
        agents = _ordered_agents(sub)
        plot_df = sub.set_index("agent_id").reindex(agents)[phase_cols].fillna(0).rename(columns=rename)
        x = np.arange(len(plot_df))
        bottom = np.zeros(len(plot_df))
        for phase in phase_labels:
            vals = plot_df[phase].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, label=phase, color=phase_colors.get(phase), alpha=0.95)
            bottom += vals
        ax.set_title(f"Average Energy Spent per Round by Phase ({exp})")
        ax.set_ylabel("Energy Spent per Round")
        ax.set_ylim(0, y_max if y_max > 0 else 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    handles, legend_labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=max(1, len(legend_labels)), bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
    _save(fig, out_dir / "01_energy_spent_by_phase_stacked.png")


def plot_category_difficulty_stacked(folders: list[Path], labels: list[str], out_dir: Path, top_categories: int = 12) -> None:
    df = _load_metric(folders, labels, "category_difficulty_attempts.csv")
    if df.empty:
        return

    tmp = df.copy()
    tmp["experiment_agent"] = _experiment_agent_col(tmp)
    top = tmp.groupby("job_category")["avg_count_per_run"].sum().sort_values(ascending=False).head(top_categories).index
    tmp = tmp[tmp["job_category"].isin(top)].copy()
    difficulties = _ordered_values(tmp["job_difficulty"], DIFFICULTY_ORDER)
    colors = _difficulty_color_map(difficulties)
    panels = list(tmp["experiment_agent"].drop_duplicates())
    if not panels:
        return

    category_order = tmp.groupby("job_category")["avg_count_per_run"].sum().sort_values(ascending=True).index.tolist()
    fig, axes = plt.subplots(1, len(panels), figsize=(max(5 * len(panels), 9), max(4, 0.42 * len(top) + 2)), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        sub = tmp[tmp["experiment_agent"] == panel]
        pivot = sub.pivot_table(index="job_category", columns="job_difficulty", values="avg_count_per_run", aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(index=category_order, columns=difficulties, fill_value=0)
        left = np.zeros(len(pivot))
        y = np.arange(len(pivot))
        for diff in difficulties:
            vals = pivot[diff].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, label=str(diff).capitalize(), color=colors[diff], alpha=0.9)
            left += vals
        ax.set_title(panel, fontsize=9)
        ax.set_xlabel("Avg attempts/run")
        ax.grid(axis="x", alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].set_yticks(np.arange(len(category_order)))
    axes[0].set_yticklabels(category_order)
    axes[0].set_ylabel("Category")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=max(1, len(difficulties)), bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Jobs Chosen by Category and Difficulty", y=1.02)
    _save(fig, out_dir / "02_jobs_chosen_category_difficulty_stacked.png")


def plot_difficulty_distribution_with_success(folders: list[Path], labels: list[str], out_dir: Path) -> None:
    diff_df = _load_metric(folders, labels, "attempted_difficulty_distribution.csv")
    success_df = _load_metric(folders, labels, "success_rate_by_difficulty.csv")
    if diff_df.empty or success_df.empty:
        return

    tmp = diff_df.copy()
    success = success_df.copy()
    merge_keys = ["experiment", "agent_id", "model_name", "job_difficulty"]
    tmp = tmp.merge(success[merge_keys + ["avg_success_rate_per_run"]], on=merge_keys, how="left")
    tmp["avg_frequency_per_run"] = tmp["avg_frequency_per_run"].fillna(0.0)
    tmp["avg_success_rate_per_run"] = tmp["avg_success_rate_per_run"].fillna(0.0).clip(0.0, 1.0)
    tmp["success_share_of_all_attempts"] = tmp["avg_frequency_per_run"] * tmp["avg_success_rate_per_run"]

    difficulties = _ordered_values(tmp["job_difficulty"], DIFFICULTY_ORDER)
    colors = _difficulty_color_map(difficulties)
    width = 0.72
    difficulty_handles = [Patch(facecolor=colors[d], edgecolor="white", label=str(d).capitalize()) for d in difficulties]

    experiments = list(tmp["experiment"].dropna().unique())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(9.5, max(4.0, 3.2 * len(experiments))), sharey=True, squeeze=False)

    for ax, exp in zip(axes.flatten(), experiments):
        sub = tmp[tmp["experiment"] == exp].copy()
        agents = _ordered_agents(sub)
        sub_share = sub.pivot_table(index="agent_id", columns="job_difficulty", values="avg_frequency_per_run", aggfunc="mean", fill_value=0)
        sub_success = sub.pivot_table(index="agent_id", columns="job_difficulty", values="success_share_of_all_attempts", aggfunc="mean", fill_value=0)
        sub_counts = sub.groupby("agent_id")["avg_count_per_run"].sum()
        sub_share = sub_share.reindex(index=agents, columns=difficulties, fill_value=0)
        sub_success = sub_success.reindex(index=agents, columns=difficulties, fill_value=0)

        x = np.arange(len(sub_share))
        bottom = np.zeros(len(sub_share))
        for diff in difficulties:
            heights = sub_share[diff].to_numpy(dtype=float)
            succeeded = np.minimum(sub_success[diff].to_numpy(dtype=float), heights)
            hatch = "\\\\\\\\" if str(diff).lower() == "medium" else "////"
            ax.bar(x, heights, bottom=bottom, width=width, color=colors[diff], edgecolor="none", label=str(diff).capitalize())
            ax.bar(x, succeeded, bottom=bottom, width=width, facecolor="none", edgecolor=_lighten_color(colors[diff], 0.58), linewidth=0.0, hatch=hatch)
            bottom += heights

        for xi, agent in enumerate(sub_share.index):
            total = sub_counts.get(agent, np.nan)
            ax.text(xi, 1.035, f"n={total:.1f}" if pd.notna(total) else "n=-", va="bottom", ha="center", fontsize=8)

        ax.set_title(f"Job Difficulty Distribution and Success Share ({exp})")
        ax.set_ylabel("Share of attempted jobs")
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(sub_share.index, rotation=30, ha="right")
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.25)

    hatch_handle = Patch(facecolor="white", edgecolor="#999999", hatch="////", label="Succeeded")
    fig.legend(handles=difficulty_handles + [hatch_handle], fontsize=8, loc="lower center", ncol=len(difficulty_handles) + 1, bbox_to_anchor=(0.5, -0.025), frameon=False)
    _save(fig, out_dir / "03_difficulty_success_hatched_overlay.png")


def make_plots(aggregate_dirs: Iterable[str | Path], labels: Iterable[str], out_dir: str | Path, top_categories: int = 12) -> None:
    folders = [Path(p) for p in aggregate_dirs]
    label_list = list(labels)
    if len(label_list) != len(folders):
        raise ValueError("Number of labels must match number of aggregate directories.")

    out = Path(out_dir)
    _ensure_out(out)
    plot_energy_over_time_aggregate(folders, label_list, out)
    plot_energy_spent(folders, label_list, out)
    plot_category_difficulty_stacked(folders, label_list, out, top_categories=top_categories)
    plot_difficulty_distribution_with_success(folders, label_list, out)


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    aggregate_dir1 = root / "comp_with_discussion" / "exponent_0.0" / "aggregate"
    aggregate_dir2 = root / "coop" / "exponent_0.0" / "aggregate"
    out = root / "plots" / "coop_vs_comp_0.0"
    make_plots([aggregate_dir1, aggregate_dir2], ["comp", "coop"], out, 12)


if __name__ == "__main__":
    main()