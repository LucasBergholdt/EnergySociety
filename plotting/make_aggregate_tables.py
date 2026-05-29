# ------------------------------------------------------------------------
# Makes aggregate tables.
# Run after analyse.py
# ------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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



# --------------------- Formatting helpers ---------------------

def _fmt_num(x, digits: int = 2) -> str:
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _fmt_pct(x, digits: int = 0) -> str:
    if pd.isna(x):
        return "-"
    try:
        return f"{100 * float(x):.{digits}f}%"
    except Exception:
        return str(x)


def _fmt_mean_std(mean, std, digits: int = 2) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return _fmt_num(mean, digits)
    return f"{_fmt_num(mean, digits)} ± {_fmt_num(std, digits)}"


def _fmt_pct_mean_std(mean, std, digits: int = 0) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return _fmt_pct(mean, digits)
    return f"{_fmt_pct(mean, digits)} ± {_fmt_pct(std, digits)}"


def _agent_sort_key(agent_id: str):
    s = str(agent_id)
    tail = s.split("_")[-1]
    return int(tail) if tail.isdigit() else s


def _save_table_png(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    font_size: int = 8,
    cell_height: float = 0.38,
    width_per_col: float = 1.35,
    max_col_width: int = 34,
) -> None:
    """Render a DataFrame as a PNG table."""
    if df.empty:
        print(f"[skip] {title}: no data")
        return

    display = df.copy()
    for col in display.columns:
        display[col] = display[col].astype(str).map(lambda x: x if len(x) <= max_col_width else x[: max_col_width - 1] + "…")

    n_rows, n_cols = display.shape
    fig_w = max(8, min(24, n_cols * width_per_col))
    fig_h = max(2.2, min(35, 1.2 + (n_rows + 1) * cell_height))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=font_size + 4, fontweight="bold", pad=12)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f8f8")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------- Metric extraction helpers ---------------------

def make_experiment_overview(folders: list[Path], labels: list[str]) -> pd.DataFrame:
    """Includes total number of collisions."""
    collisions = _load_metric(folders, labels, "average_collisions.csv")
    rows = []

    if not collisions.empty:
        for exp, grp in collisions.groupby("experiment"):
            row = {"experiment": exp}
            for _, r in grp.iterrows():
                metric = str(r["metric"])
                row[metric] = _fmt_mean_std(r.get("mean"), r.get("std"), 2)
                row[f"{metric}_range"] = f"{_fmt_num(r.get('min'), 0)}-{_fmt_num(r.get('max'), 0)}"
            rows.append(row)

    out = pd.DataFrame(rows)
    wanted = ["experiment", "total_collisions", "total_collisions_range"]
    return out[wanted] if not out.empty else out


def _base_agent_index(folders: list[Path], labels: list[str]) -> pd.DataFrame:
    base = _load_metric(folders, labels, "energy_gained_per_spent.csv")
    base = base[["experiment", "agent_id", "model_name"]].drop_duplicates()
    base = base.sort_values(["experiment", "agent_id"], key=lambda s: s.map(_agent_sort_key) if s.name == "agent_id" else s)
    return base


def make_agent_summary(folders: list[Path], labels: list[str]) -> pd.DataFrame:
    """
    Includes:
      - Energy gained per energy spent
      - Average rounds active
      - Time to first deactivation
      - Donations received
      - Donations made and average donation amount
      - Average energy spent per phase and per round
    """
    base = _base_agent_index(folders, labels)
    if base.empty:
        return base
    out = base.copy()

    eff = _load_metric(folders, labels, "energy_gained_per_spent.csv")
    if not eff.empty:
        eff_rows = []
        for _, r in eff.iterrows():
            eff_rows.append({
                "experiment": r["experiment"],
                "agent_id": r["agent_id"],
                "gained/spent": _fmt_mean_std(r.get("avg_gained_per_spent"), r.get("std_gained_per_spent"), 2),
                "avg reward": _fmt_num(r.get("avg_energy_rewarded"), 1),
                "avg spent": _fmt_num(r.get("avg_energy_spent"), 1),
            })
        out = out.merge(pd.DataFrame(eff_rows), on=["experiment", "agent_id"], how="left")

    active = _load_metric(folders, labels, "rounds_active.csv")
    if not active.empty:
        active_rows = []
        for _, r in active.iterrows():
            active_rows.append({
                "experiment": r["experiment"],
                "agent_id": r["agent_id"],
                "rounds active": _fmt_mean_std(r.get("avg_rounds_active"), r.get("std_rounds_active"), 1),
            })
        out = out.merge(pd.DataFrame(active_rows), on=["experiment", "agent_id"], how="left")

    deact = _load_metric(folders, labels, "first_deactivation.csv")
    if not deact.empty:
        deact_rows = []
        for _, r in deact.iterrows():
            deact_rows.append({
                "experiment": r["experiment"],
                "agent_id": r["agent_id"],
                "deact. rate": _fmt_pct(r.get("deactivation_rate"), 0),
            })
        out = out.merge(pd.DataFrame(deact_rows), on=["experiment", "agent_id"], how="left")

    dons = _load_metric(folders, labels, "donations_received.csv")
    if not dons.empty:
        don_rows = []
        for _, r in dons.iterrows():
            don_rows.append({
                "experiment": r["experiment"],
                "agent_id": r["agent_id"],
                "donations received/run": _fmt_mean_std(r.get("avg_donations_received"), r.get("std_donations_received"), 1),
                "donations total": _fmt_num(r.get("total_donations_received"), 0),
            })
        out = out.merge(pd.DataFrame(don_rows), on=["experiment", "agent_id"], how="left")

    made = _load_metric(folders, labels, "donations_made.csv")
    if not made.empty:
        made_rows = []
        for _, r in made.iterrows():
            made_rows.append({
                "experiment": r["experiment"],
                "agent_id": r["agent_id"],
                "donations made/run": _fmt_mean_std(r.get("avg_donations_made"), r.get("std_donations_made"), 1),
                "energy donated/run": _fmt_mean_std(r.get("avg_energy_donated_per_run"), r.get("std_energy_donated_per_run"), 1),
                "energy/donation": _fmt_mean_std(r.get("avg_energy_per_donation"), r.get("std_energy_per_donation"), 1),
                "energy donated total": _fmt_num(r.get("total_energy_donated"), 0),
            })
        out = out.merge(pd.DataFrame(made_rows), on=["experiment", "agent_id"], how="left")

    energy = _load_metric(folders, labels, "energy_spent_per_phase.csv")
    if not energy.empty:
        energy_rows = []
        for _, r in energy.iterrows():
            def ms(prefix: str) -> str:
                return _fmt_mean_std(r.get(f"{prefix}_mean"), r.get(f"{prefix}_std"), 1)

            energy_rows.append({
                "experiment": r["experiment"],
                "agent_id": r["agent_id"],
                "energy vote/round": ms("energy_vote"),
                "energy decide/round": ms("energy_decide"),
                "energy act/round": ms("energy_act"),
                "energy total/round": ms("total_energy_spent"),
            })
        out = out.merge(pd.DataFrame(energy_rows), on=["experiment", "agent_id"], how="left")

    return out.fillna("-")


def make_action_frequency_table(folders: list[Path], labels: list[str]) -> pd.DataFrame:
    """Includes chosen action frequency."""
    df = _load_metric(folders, labels, "action_frequency.csv")
    if df.empty:
        return df

    pivot = df.pivot_table(index=["experiment", "agent_id", "model_name"], columns="action", values="avg_frequency_per_run", aggfunc="mean").reset_index()
    for c in pivot.columns:
        if c not in ["experiment", "agent_id", "model_name"]:
            pivot[c] = pivot[c].map(lambda x: _fmt_pct(x, 0))
    action_order = ["ATTEMPT_JOB", "IDLE", "DONATE", "SABOTAGE_JOB"]
    cols = ["experiment", "agent_id", "model_name"] + [c for c in action_order if c in pivot.columns]
    return pivot[cols].sort_values(["experiment", "agent_id"], key=lambda s: s.map(_agent_sort_key) if s.name == "agent_id" else s).fillna("-")


def make_error_count_table(folders: list[Path], labels: list[str]) -> pd.DataFrame:
    """Includes only the total ERROR count."""
    df = _load_metric(folders, labels, "action_frequency_with_errors.csv")
    if df.empty:
        return df

    errors = df[df["action"].astype(str) == "ERROR"].copy()
    if errors.empty:
        return pd.DataFrame()

    out = errors[["experiment", "agent_id", "model_name", "count"]].rename(columns={"count": "ERROR count"})
    out["ERROR count"] = out["ERROR count"].map(lambda x: _fmt_num(x, 0))
    return out.sort_values(["experiment", "agent_id"], key=lambda s: s.map(_agent_sort_key) if s.name == "agent_id" else s).fillna("-")


def make_job_attempt_distribution_table(folders: list[Path], labels: list[str]) -> pd.DataFrame:
    """Includes job attempt distribution by difficulty."""
    df = _load_metric(folders, labels, "attempted_difficulty_distribution.csv")
    if df.empty:
        return df

    pivot = df.pivot_table(index=["experiment", "agent_id", "model_name"], columns="job_difficulty", values="avg_count_per_run", aggfunc="mean").reset_index()
    for c in pivot.columns:
        if c not in ["experiment", "agent_id", "model_name"]:
            pivot[c] = pivot[c].map(lambda x: _fmt_num(x, 1))
    diff_order = ["easy", "medium", "hard"]
    cols = ["experiment", "agent_id", "model_name"] + [c for c in diff_order if c in pivot.columns]
    return pivot[cols].sort_values(["experiment", "agent_id"], key=lambda s: s.map(_agent_sort_key) if s.name == "agent_id" else s).fillna("-")


def make_success_rate_table(folders: list[Path], labels: list[str]) -> pd.DataFrame:
    """Includes success rate per difficulty."""
    df = _load_metric(folders, labels, "success_rate_by_difficulty.csv")
    if df.empty:
        return df

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "experiment": r["experiment"],
            "agent_id": r["agent_id"],
            "model_name": r.get("model_name", "-"),
            "difficulty": r["job_difficulty"],
            "value": _fmt_pct_mean_std(r.get("avg_success_rate_per_run"), r.get("std_success_rate_per_run"), 0),
        })
    tmp = pd.DataFrame(rows)
    pivot = tmp.pivot_table(index=["experiment", "agent_id", "model_name"], columns="difficulty", values="value", aggfunc="first").reset_index()
    diff_order = ["easy", "medium", "hard"]
    cols = ["experiment", "agent_id", "model_name"] + [c for c in diff_order if c in pivot.columns]
    return pivot[cols].sort_values(["experiment", "agent_id"], key=lambda s: s.map(_agent_sort_key) if s.name == "agent_id" else s).fillna("-")



# --------------------- Main table generation ---------------------

def make_aggregate_tables(
    aggregate_dirs: Iterable[str | Path],
    labels: Iterable[str],
    out_dir: str | Path,
) -> None:
    folders = [Path(p) for p in aggregate_dirs]
    label_list = list(labels)
    if len(label_list) != len(folders):
        raise ValueError("Number of labels must match number of aggregate directories.")

    out = Path(out_dir)
    _ensure_out(out)

    overview = make_experiment_overview(folders, label_list)
    summary = make_agent_summary(folders, label_list)
    actions = make_action_frequency_table(folders, label_list)
    error_counts = make_error_count_table(folders, label_list)
    difficulty = make_job_attempt_distribution_table(folders, label_list)
    success = make_success_rate_table(folders, label_list)

    # 1. Experiment-level metrics.
    _save_table_png(
        overview,
        "Aggregate Experiment Overview",
        out / "table_01_experiment_overview.png",
        font_size=9,
        width_per_col=1.8,
    )

    # 2. Main per-agent summary. This is the densest table.
    _save_table_png(
        summary,
        "Per-Agent Aggregate Summary",
        out / "table_02_agent_summary.png",
        font_size=6,
        cell_height=0.34,
        width_per_col=1.2,
        max_col_width=24,
    )

    # 3. Behaviour and job choice.
    action_parts = []
    if not actions.empty:
        actions2 = actions.copy()
        actions2.columns = [c if c in ["experiment", "agent_id", "model_name"] else f"chosen action: {c}" for c in actions2.columns]
        action_parts.append(actions2)
    if not error_counts.empty:
        action_parts.append(error_counts)
    if not action_parts:
        action_table = pd.DataFrame()
    else:
        action_table = action_parts[0]
        for part in action_parts[1:]:
            action_table = action_table.merge(part, on=["experiment", "agent_id", "model_name"], how="outer")

    if not action_table.empty and not difficulty.empty:
        diff2 = difficulty.copy()
        diff2.columns = [c if c in ["experiment", "agent_id", "model_name"] else f"jobs: {c}" for c in diff2.columns]
        behaviour = action_table.merge(diff2, on=["experiment", "agent_id", "model_name"], how="outer").fillna("-")
    else:
        behaviour = action_table if not action_table.empty else difficulty
    _save_table_png(
        behaviour,
        "Action Frequency and Job Difficulty Distribution",
        out / "table_03_behaviour_and_difficulty.png",
        font_size=7,
        cell_height=0.36,
        width_per_col=1.2,
        max_col_width=22,
    )

    # 4. Success rates.
    performance = success
    _save_table_png(
        performance,
        "Task Success Rates",
        out / "table_04_success_and_categories.png",
        font_size=7,
        cell_height=0.38,
        width_per_col=1.45,
        max_col_width=48,
    )

    # Also write the underlying compact tables as CSVs for easy inspection/editing.
    for name, df in {
        "compact_experiment_overview": overview,
        "compact_agent_summary": summary,
        "compact_behaviour_and_difficulty": behaviour,
        "compact_success_and_categories": performance,
    }.items():
        if not df.empty:
            df.to_csv(out / f"{name}.csv", index=False)


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    aggregate_dir1 = root / "comp_with_discussion" / "exponent_0.5" / "aggregate"
    aggregate_dir2 = root / "coop" / "exponent_0.5" / "aggregate"
    out = root / "tables" / "coop_vs_comp_0.5"
    make_aggregate_tables([aggregate_dir1, aggregate_dir2], ["comp", "coop"], out)


if __name__ == "__main__":
    main()