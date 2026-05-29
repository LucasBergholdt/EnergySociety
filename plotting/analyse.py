# ------------------------------------------------------------------------
# This file is purely for gathering statistcs used by plotting scripts.
# ------------------------------------------------------------------------

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

DIFFICULTY_ORDER = ["easy", "medium", "hard"]
ACTION_ORDER = ["ATTEMPT_JOB", "IDLE", "DONATE", "SABOTAGE_JOB", "INACTIVE"]
ACTIVE_ACTION_ORDER = ["ATTEMPT_JOB", "IDLE", "DONATE", "SABOTAGE_JOB"]
ACTION_WITH_ERROR_ORDER = ACTIVE_ACTION_ORDER + ["ERROR"]


def find_run_dirs(experiment_dir: str | Path) -> list[Path]:
    root = Path(experiment_dir)
    if not root.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {root}")
    run_dirs = sorted({p.parent for p in root.rglob("agents.json") if (p.parent / "config.json").exists()})
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with agents.json/config.json found under: {root}")
    return run_dirs


def load_run(run_dir: str | Path) -> tuple[list, dict]:
    run_dir = Path(run_dir)
    with open(run_dir / "agents.json") as f:
        agents = json.load(f)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    return agents, config


def _run_label(run_dir: Path, config: dict, fallback_idx: int) -> str:
    if config.get("run_number") is not None:
        return f"run_{config['run_number']}"
    if config.get("seed") is not None:
        return f"seed_{config['seed']}"
    return run_dir.name or f"run_{fallback_idx}"


def build_memory_df(agents: list) -> pd.DataFrame:
    error_lookup: dict[tuple, str] = {}
    for agent in agents:
        for err in agent.get("errors", []):
            error_lookup[(agent["agent_id"], err["round_num"], err["phase"])] = err["error_type"]

    rows = []
    for agent in agents:
        for entry in agent.get("memory", []):
            job = entry.get("attempted_job") or {}
            agent_id = agent["agent_id"]
            round_num = entry["round_num"]
            decide_error = error_lookup.get((agent_id, round_num, "decide"))
            act_error = error_lookup.get((agent_id, round_num, "act"))
            vote_error = error_lookup.get((agent_id, round_num, "vote"))
            rows.append({
                "agent_id": agent_id,
                "model_name": agent["model_name"],
                "model_size": agent["model_size"],
                "round_num": round_num,
                "action": entry.get("action"),
                "energy_start": entry.get("energy_at_start_of_round", 0.0),
                "energy_vote": entry.get("energy_spent_voting") or 0.0,
                "energy_decide": entry.get("energy_spent_deciding") or 0.0,
                "energy_act": entry.get("energy_spent_on_action") or 0.0,
                "energy_rewarded": entry.get("energy_rewarded") or 0.0,
                "succeeded": entry.get("succeeded"),
                "job_difficulty": job.get("difficulty"),
                "job_category": job.get("category"),
                "job_id": job.get("job_id"),
                "job_reward": job.get("reward"),
                "target_agent": entry.get("target_agent"),
                "donated_amount": entry.get("donated_amount"),
                "decide_error": decide_error,
                "act_error": act_error,
                "vote_error": vote_error,
                "any_error": any(v is not None for v in [decide_error, act_error, vote_error]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["total_energy_spent"] = df["energy_vote"] + df["energy_decide"] + df["energy_act"]
    df["net_energy"] = df["energy_rewarded"] - df["total_energy_spent"]
    df["energy_end"] = df["energy_start"] + df["net_energy"]
    return df


def build_agent_df(agents: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "agent_id": a["agent_id"],
        "model_name": a["model_name"],
        "model_size": a["model_size"],
        "initial_energy": a["init_energy"],
        "final_energy": a["final_energy"],
        "rounds_active": a["rounds_active"],
        "deactivated_at": a["deactivated_at_round"],
        "survived": a["final_energy"] > 0,
    } for a in agents])


def load_experiment_runs(experiment_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mem_frames, agent_frames, run_rows = [], [], []
    for idx, run_dir in enumerate(find_run_dirs(experiment_dir), start=1):
        agents, config = load_run(run_dir)
        run_id = _run_label(run_dir, config, idx)
        experiment = config.get("experiment", Path(experiment_dir).name)
        seed = config.get("seed")
        max_rounds = config.get("max_rounds")

        mem_df = build_memory_df(agents)
        agent_df = build_agent_df(agents)

        for df in [mem_df, agent_df]:
            if not df.empty:
                df["run_id"] = run_id
                df["run_dir"] = str(run_dir)
                df["experiment"] = experiment
                df["seed"] = seed
                df["max_rounds"] = max_rounds

        mem_frames.append(mem_df)
        agent_frames.append(agent_df)
        run_rows.append({
            "run_id": run_id,
            "run_dir": str(run_dir),
            "experiment": experiment,
            "seed": seed,
            "max_rounds": max_rounds,
            "num_jobs": config.get("num_jobs"),
            "model_size_exponent": config.get("model_size_exponent"),
            "prompt_mode": config.get("prompt_mode"),
            "use_votes": config.get("use_votes"),
            "enable_sabotage": config.get("enable_sabotage"),
        })

    return (
        pd.concat(mem_frames, ignore_index=True) if mem_frames else pd.DataFrame(),
        pd.concat(agent_frames, ignore_index=True) if agent_frames else pd.DataFrame(),
        pd.DataFrame(run_rows),
    )


def _attempts(memory_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty:
        return memory_df.copy()
    return memory_df[(memory_df["action"] == "ATTEMPT_JOB") & memory_df["job_id"].notna()].copy()


def load_transcript(run_dir: str | Path) -> list:
    path = Path(run_dir) / "transcript.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def build_vote_recommendation_df(transcript: list, agents: list | None = None) -> pd.DataFrame:
    model_lookup = {a.get("agent_id"): a.get("model_name") for a in agents or []}
    rows = []
    for entry in transcript or []:
        if entry.get("phase") != "vote":
            continue
        structured = entry.get("structured_response") or {}
        voter_id = structured.get("voter_id") or entry.get("agent_id")
        for rec_idx, rec in enumerate(structured.get("recommendations") or [], start=1):
            recommended_agent_id = rec.get("agent_id")
            rows.append({
                "round_num": entry.get("round_num"),
                "voter_agent_id": voter_id,
                "voter_model_name": model_lookup.get(voter_id),
                "recommended_agent_id": recommended_agent_id,
                "recommended_model_name": model_lookup.get(recommended_agent_id),
                "recommendation_index": rec_idx,
                "recommended_action": rec.get("recommended_action"),
                "job_id": rec.get("job_id"),
                "target_agent": rec.get("target_agent"),
                "amount": rec.get("amount"),
                "target_type": "self" if voter_id == recommended_agent_id else "other",
            })
    cols = [
        "round_num", "voter_agent_id", "voter_model_name", "recommended_agent_id",
        "recommended_model_name", "recommendation_index", "recommended_action",
        "job_id", "target_agent", "amount", "target_type",
    ]
    return pd.DataFrame(rows, columns=cols)


def load_experiment_recommendations(experiment_dir: str | Path) -> pd.DataFrame:
    frames = []
    for idx, run_dir in enumerate(find_run_dirs(experiment_dir), start=1):
        agents, config = load_run(run_dir)
        vote_df = build_vote_recommendation_df(load_transcript(run_dir), agents)
        if vote_df.empty:
            continue
        vote_df["run_id"] = _run_label(run_dir, config, idx)
        vote_df["run_dir"] = str(run_dir)
        vote_df["experiment"] = config.get("experiment", Path(experiment_dir).name)
        vote_df["seed"] = config.get("seed")
        vote_df["max_rounds"] = config.get("max_rounds")
        frames.append(vote_df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=[
        "round_num", "voter_agent_id", "voter_model_name", "recommended_agent_id",
        "recommended_model_name", "recommendation_index", "recommended_action",
        "job_id", "target_agent", "amount", "target_type", "run_id", "run_dir",
        "experiment", "seed", "max_rounds",
    ])


def metric_recommended_job_collisions(vote_df: pd.DataFrame) -> pd.DataFrame:
    return _recommended_collision_metric(vote_df, mode="all")


def metric_recommended_other_only_job_collisions(vote_df: pd.DataFrame) -> pd.DataFrame:
    return _recommended_collision_metric(vote_df, mode="other_only")


def _recommended_collision_metric(vote_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if vote_df.empty:
        return pd.DataFrame()
    attempts = vote_df[(vote_df["recommended_action"] == "ATTEMPT_JOB") & vote_df["job_id"].notna()].copy()
    if attempts.empty:
        return pd.DataFrame()

    group_cols = ["run_id", "voter_agent_id", "voter_model_name", "round_num", "job_id"]
    if mode == "other_only":
        attempts["is_voter_recommendation"] = attempts["recommended_agent_id"] == attempts["voter_agent_id"]
        attempts["other_recommended_agent_id"] = attempts["recommended_agent_id"].where(
            ~attempts["is_voter_recommendation"]
        )
        per_job = attempts.groupby(group_cols).agg(
            num_agents_recommended=("other_recommended_agent_id", "nunique"),
            voter_included=("is_voter_recommendation", "any"),
        ).reset_index()
        collisions = per_job[(per_job["num_agents_recommended"] > 1) & (~per_job["voter_included"])].copy()
    else:
        per_job = attempts.groupby(group_cols).agg(num_agents_recommended=("recommended_agent_id", "nunique")).reset_index()
        collisions = per_job[per_job["num_agents_recommended"] > 1].copy()
    base = vote_df[["run_id", "voter_agent_id", "voter_model_name"]].drop_duplicates()

    prefix = {
        "all": "",
        "other_only": "other_only_",
    }[mode]
    event_col = f"{prefix}collision_recommendation_events"
    rounds_col = f"{prefix}rounds_with_collision_recommendation"
    excess_col = f"{prefix}excess_agents_assigned_to_colliding_jobs"

    if collisions.empty:
        per_run = base.copy()
        per_run[event_col] = 0
        per_run[rounds_col] = 0
        per_run[excess_col] = 0
    else:
        collisions["excess_agents"] = collisions["num_agents_recommended"] - 1
        counts = collisions.groupby(["run_id", "voter_agent_id", "voter_model_name"], as_index=False).agg(
            **{event_col: ("job_id", "size"), rounds_col: ("round_num", "nunique"), excess_col: ("excess_agents", "sum")}
        )
        per_run = base.merge(counts, on=["run_id", "voter_agent_id", "voter_model_name"], how="left")
        for col in [event_col, rounds_col, excess_col]:
            per_run[col] = per_run[col].fillna(0).astype(int)

    return per_run.groupby(["voter_agent_id", "voter_model_name"], as_index=False).agg(
        **{
            f"avg_{event_col}": (event_col, "mean"),
            f"total_{event_col}": (event_col, "sum"),
            f"avg_{rounds_col}": (rounds_col, "mean"),
            f"avg_{excess_col}": (excess_col, "mean"),
            "n_runs": ("run_id", "nunique"),
        }
    ).sort_values("voter_agent_id")


def metric_recommended_donations_to_self(vote_df: pd.DataFrame) -> pd.DataFrame:
    if vote_df.empty:
        return pd.DataFrame()
    donations = vote_df[
        (vote_df["recommended_action"] == "DONATE") &
        (vote_df["target_agent"] == vote_df["voter_agent_id"]) &
        (vote_df["recommended_agent_id"] != vote_df["voter_agent_id"])
    ].copy()
    base = vote_df[["run_id", "voter_agent_id", "voter_model_name"]].drop_duplicates()
    per_run = donations.groupby(["run_id", "voter_agent_id", "voter_model_name"], as_index=False).agg(
        donation_to_self_recommendations=("recommended_agent_id", "size"),
        requested_energy_to_self=("amount", "sum"),
    )
    per_run = base.merge(per_run, on=["run_id", "voter_agent_id", "voter_model_name"], how="left").fillna({
        "donation_to_self_recommendations": 0,
        "requested_energy_to_self": 0,
    })
    return per_run.groupby(["voter_agent_id", "voter_model_name"], as_index=False).agg(
        avg_donation_to_self_recommendations=("donation_to_self_recommendations", "mean"),
        total_donation_to_self_recommendations=("donation_to_self_recommendations", "sum"),
        avg_requested_energy_to_self=("requested_energy_to_self", "mean"),
        total_requested_energy_to_self=("requested_energy_to_self", "sum"),
        n_runs=("run_id", "nunique"),
    ).sort_values("voter_agent_id")


def metric_recommendation_action_self_vs_others(vote_df: pd.DataFrame) -> pd.DataFrame:
    if vote_df.empty:
        return pd.DataFrame()
    keys = ["voter_agent_id", "voter_model_name", "run_id", "target_type"]
    actions = pd.DataFrame({"recommended_action": ACTION_ORDER})
    base = vote_df[keys].drop_duplicates().merge(actions, how="cross")
    counts = vote_df.groupby(keys + ["recommended_action"]).size().reset_index(name="count_per_run")
    counts = base.merge(counts, on=keys + ["recommended_action"], how="left").fillna({"count_per_run": 0})
    totals = counts.groupby(keys)["count_per_run"].transform("sum")
    counts["frequency_per_run"] = counts["count_per_run"] / totals.replace(0, np.nan)
    return counts.groupby(["voter_agent_id", "voter_model_name", "target_type", "recommended_action"], as_index=False).agg(
        total_count=("count_per_run", "sum"),
        avg_count_per_run=("count_per_run", "mean"),
        avg_frequency_per_run=("frequency_per_run", "mean"),
        n_runs=("run_id", "nunique"),
    )


def _recommendation_matches_actual(rec: pd.Series) -> bool:
    action = rec.get("recommended_action")
    if action != rec.get("action"):
        return False
    if action in ["ATTEMPT_JOB", "SABOTAGE_JOB"]:
        return rec.get("job_id_rec") == rec.get("job_id_actual")
    if action == "DONATE":
        return rec.get("target_agent_rec") == rec.get("target_agent_actual")
    return action == "IDLE"


def build_recommendation_enactment_df(vote_df: pd.DataFrame, memory_df: pd.DataFrame) -> pd.DataFrame:
    if vote_df.empty or memory_df.empty:
        return pd.DataFrame()
    actual = memory_df[[
        "run_id", "round_num", "agent_id", "action", "job_id", "target_agent",
        "energy_rewarded", "total_energy_spent", "net_energy",
    ]].rename(columns={
        "agent_id": "recommended_agent_id",
        "job_id": "job_id_actual",
        "target_agent": "target_agent_actual",
    })
    recs = vote_df.rename(columns={"job_id": "job_id_rec", "target_agent": "target_agent_rec"})
    out = recs.merge(actual, on=["run_id", "round_num", "recommended_agent_id"], how="left")
    out["has_actual_decision"] = out["action"].notna()
    out["strict_followed"] = out.apply(_recommendation_matches_actual, axis=1)
    out["action_only_followed"] = out["recommended_action"] == out["action"]
    return out


def metric_recommendation_follow_rate(vote_df: pd.DataFrame, memory_df: pd.DataFrame) -> pd.DataFrame:
    enacted = build_recommendation_enactment_df(vote_df, memory_df)
    if enacted.empty:
        return pd.DataFrame()
    enacted = enacted[enacted["has_actual_decision"]].copy()
    if enacted.empty:
        return pd.DataFrame()
    return enacted.groupby(["voter_agent_id", "voter_model_name", "target_type", "recommended_action"], as_index=False).agg(
        total_recommendations=("recommended_agent_id", "size"),
        strict_followed=("strict_followed", "sum"),
        action_only_followed=("action_only_followed", "sum"),
    ).assign(
        strict_follow_rate=lambda d: d["strict_followed"] / d["total_recommendations"].replace(0, np.nan),
        action_only_follow_rate=lambda d: d["action_only_followed"] / d["total_recommendations"].replace(0, np.nan),
    )


def metric_recommendation_decision_follow_source(vote_df: pd.DataFrame, memory_df: pd.DataFrame) -> pd.DataFrame:
    if vote_df.empty or memory_df.empty:
        return pd.DataFrame()

    actual = memory_df[[
        "run_id", "round_num", "agent_id", "model_name", "action", "job_id", "target_agent",
    ]].copy().rename(columns={
        "agent_id": "recommended_agent_id",
        "job_id": "job_id_actual",
        "target_agent": "target_agent_actual",
    })
    recs = vote_df.rename(columns={"job_id": "job_id_rec", "target_agent": "target_agent_rec"})
    joined = actual.merge(recs, on=["run_id", "round_num", "recommended_agent_id"], how="left")
    joined["strict_followed"] = joined.apply(_recommendation_matches_actual, axis=1)
    joined["is_own_recommendation"] = joined["voter_agent_id"] == joined["recommended_agent_id"]

    decision = joined.groupby(["run_id", "round_num", "recommended_agent_id", "model_name"], as_index=False).agg(
        own_followed=("strict_followed", lambda s: bool(s[joined.loc[s.index, "is_own_recommendation"]].any())),
        any_other_followed=("strict_followed", lambda s: bool(s[~joined.loc[s.index, "is_own_recommendation"]].any())),
    )
    decision["other_followed"] = decision["any_other_followed"] & ~decision["own_followed"]
    decision["no_recommendation_followed"] = ~decision["own_followed"] & ~decision["any_other_followed"]

    for col in ["own_followed", "other_followed", "no_recommendation_followed"]:
        decision[f"{col}_int"] = decision[col].astype(int)

    return decision.groupby(["recommended_agent_id", "model_name"], as_index=False).agg(
        own_followed_decisions=("own_followed_int", "sum"),
        other_followed_decisions=("other_followed_int", "sum"),
        no_recommendation_followed_decisions=("no_recommendation_followed_int", "sum"),
        total_decisions=("round_num", "size"),
    ).assign(
        own_follow_rate=lambda d: d["own_followed_decisions"] / d["total_decisions"].replace(0, np.nan),
        other_follow_rate=lambda d: d["other_followed_decisions"] / d["total_decisions"].replace(0, np.nan),
        no_recommendation_follow_rate=lambda d: d["no_recommendation_followed_decisions"] / d["total_decisions"].replace(0, np.nan),
    ).rename(columns={"recommended_agent_id": "agent_id"})


def metric_enacted_recommendation_net_energy(vote_df: pd.DataFrame, memory_df: pd.DataFrame) -> pd.DataFrame:
    enacted = build_recommendation_enactment_df(vote_df, memory_df)
    if enacted.empty:
        return pd.DataFrame()
    followed = enacted[enacted["strict_followed"]].copy()
    if followed.empty:
        return pd.DataFrame()
    return followed.groupby(["voter_agent_id", "voter_model_name", "target_type", "recommended_action"], as_index=False).agg(
        enacted_recommendations=("recommended_agent_id", "size"),
        avg_actual_net_energy=("net_energy", "mean"),
        total_actual_net_energy=("net_energy", "sum"),
    )


def metric_average_collisions(memory_df: pd.DataFrame) -> pd.DataFrame:
    attempts = _attempts(memory_df)
    if attempts.empty:
        return pd.DataFrame()
    per_job = attempts.groupby(["run_id", "round_num", "job_id"]).agg(num_agents=("agent_id", "nunique")).reset_index()
    collisions = per_job[per_job["num_agents"] > 1]
    per_run = collisions.groupby("run_id").agg(total_collisions=("job_id", "size")).reset_index()
    all_runs = memory_df[["run_id"]].drop_duplicates()
    per_run = all_runs.merge(per_run, on="run_id", how="left").fillna(0)
    return pd.DataFrame({
        "metric": ["total_collisions"],
        "mean": [per_run["total_collisions"].mean()],
        "std": [per_run["total_collisions"].std()],
        "min": [per_run["total_collisions"].min()],
        "max": [per_run["total_collisions"].max()],
    })


def metric_energy_gained_per_spent(memory_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty:
        return pd.DataFrame()
    per_run = memory_df.groupby(["agent_id", "model_name", "run_id"], as_index=False).agg(
        energy_gained=("energy_rewarded", "sum"),
        energy_spent=("total_energy_spent", "sum"),
    )
    per_run["gained_per_spent"] = per_run["energy_gained"] / per_run["energy_spent"].replace(0, np.nan)
    return per_run.groupby(["agent_id", "model_name"], as_index=False).agg(
        avg_gained_per_spent=("gained_per_spent", "mean"),
        std_gained_per_spent=("gained_per_spent", "std"),
        avg_energy_rewarded=("energy_gained", "mean"),
        avg_energy_spent=("energy_spent", "mean"),
        n_runs=("run_id", "nunique"),
    )


def metric_rounds_active(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()
    return agent_df.groupby(["agent_id", "model_name"], as_index=False).agg(
        avg_rounds_active=("rounds_active", "mean"),
        std_rounds_active=("rounds_active", "std"),
        n_runs=("run_id", "nunique"),
    )


def metric_first_deactivation(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()
    df = agent_df.copy()
    df["deactivated"] = df["deactivated_at"].notna()
    return df.groupby(["agent_id", "model_name"], as_index=False).agg(
        deactivation_rate=("deactivated", "mean"),
        n_runs=("run_id", "nunique"),
    )


def metric_donations_received(memory_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty:
        return pd.DataFrame()
    donations = memory_df[memory_df["target_agent"].notna()].copy()
    if donations.empty:
        return pd.DataFrame()
    rows = []
    agent_model = memory_df[["agent_id", "model_name"]].drop_duplicates().set_index("agent_id")["model_name"].to_dict()
    for _, row in donations.iterrows():
        rows.append({
            "run_id": row["run_id"],
            "agent_id": row["target_agent"],
            "model_name": agent_model.get(row["target_agent"]),
            "amount": row["donated_amount"] or 0,
        })
    df = pd.DataFrame(rows)
    per_run = df.groupby(["agent_id", "model_name", "run_id"], as_index=False).agg(
        donations_received=("amount", "size"),
        energy_received=("amount", "sum"),
    )
    return per_run.groupby(["agent_id", "model_name"], as_index=False).agg(
        avg_donations_received=("donations_received", "mean"),
        std_donations_received=("donations_received", "std"),
        total_donations_received=("donations_received", "sum"),
        avg_energy_received_per_run=("energy_received", "mean"),
        total_energy_received=("energy_received", "sum"),
        n_runs=("run_id", "nunique"),
    )


def metric_donations_made(memory_df: pd.DataFrame) -> pd.DataFrame:
    donations = memory_df[memory_df["action"] == "DONATE"].copy() if not memory_df.empty else pd.DataFrame()
    if donations.empty:
        return pd.DataFrame()
    per_run = donations.groupby(["agent_id", "model_name", "run_id"], as_index=False).agg(
        donations_made=("donated_amount", "size"),
        energy_donated=("donated_amount", "sum"),
    )
    per_run["energy_per_donation"] = per_run["energy_donated"] / per_run["donations_made"].replace(0, np.nan)
    return per_run.groupby(["agent_id", "model_name"], as_index=False).agg(
        avg_donations_made=("donations_made", "mean"),
        std_donations_made=("donations_made", "std"),
        total_donations_made=("donations_made", "sum"),
        avg_energy_donated_per_run=("energy_donated", "mean"),
        std_energy_donated_per_run=("energy_donated", "std"),
        total_energy_donated=("energy_donated", "sum"),
        avg_energy_per_donation=("energy_per_donation", "mean"),
        std_energy_per_donation=("energy_per_donation", "std"),
        n_runs=("run_id", "nunique"),
    )


def metric_action_frequency(memory_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty:
        return pd.DataFrame()
    keys = ["agent_id", "model_name", "run_id"]
    actions = [a for a in ACTION_WITH_ERROR_ORDER if a in set(memory_df["action"].dropna())]
    actions += sorted(set(memory_df["action"].dropna()) - set(actions))
    grid = memory_df[keys].drop_duplicates().merge(pd.DataFrame({"action": actions}), how="cross")
    counts = memory_df.groupby(keys + ["action"]).size().reset_index(name="count_per_run")
    counts = grid.merge(counts, on=keys + ["action"], how="left").fillna({"count_per_run": 0})
    totals = counts.groupby(keys)["count_per_run"].transform("sum")
    counts["frequency_per_run"] = counts["count_per_run"] / totals.replace(0, np.nan)
    return counts.groupby(["agent_id", "model_name", "action"], as_index=False).agg(
        count=("count_per_run", "sum"),
        avg_count_per_run=("count_per_run", "mean"),
        avg_frequency_per_run=("frequency_per_run", "mean"),
        n_runs=("run_id", "nunique"),
    )


def metric_action_frequency_with_errors(memory_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty:
        return pd.DataFrame()
    df = memory_df.copy()
    df["action"] = np.where(df["any_error"], "ERROR", df["action"])
    return metric_action_frequency(df)


def metric_attempted_difficulty_distribution(memory_df: pd.DataFrame) -> pd.DataFrame:
    attempts = _attempts(memory_df)
    if attempts.empty:
        return pd.DataFrame()
    keys = ["agent_id", "model_name", "run_id"]
    grid = memory_df[keys].drop_duplicates().merge(pd.DataFrame({"job_difficulty": DIFFICULTY_ORDER}), how="cross")
    counts = attempts.groupby(keys + ["job_difficulty"]).size().reset_index(name="count_per_run")
    counts = grid.merge(counts, on=keys + ["job_difficulty"], how="left").fillna({"count_per_run": 0})
    totals = counts.groupby(keys)["count_per_run"].transform("sum")
    counts["frequency_per_run"] = counts["count_per_run"] / totals.replace(0, np.nan)
    return counts.groupby(["agent_id", "model_name", "job_difficulty"], as_index=False).agg(
        count=("count_per_run", "sum"),
        avg_count_per_run=("count_per_run", "mean"),
        std_count_per_run=("count_per_run", "std"),
        avg_frequency_per_run=("frequency_per_run", "mean"),
        n_runs=("run_id", "nunique"),
    )


def metric_success_rate_by_difficulty(memory_df: pd.DataFrame) -> pd.DataFrame:
    attempts = _attempts(memory_df)
    if attempts.empty:
        return pd.DataFrame()
    attempts = attempts[attempts["job_difficulty"].notna()].copy()
    attempts["success_int"] = attempts["succeeded"].eq(True).astype(int)
    pooled = attempts.groupby(["agent_id", "model_name", "job_difficulty"], as_index=False).agg(
        attempted=("success_int", "size"),
        succeeded=("success_int", "sum"),
    )
    pooled["success_rate_pooled"] = pooled["succeeded"] / pooled["attempted"]
    per_run = attempts.groupby(["agent_id", "model_name", "run_id", "job_difficulty"], as_index=False).agg(
        attempted_run=("success_int", "size"),
        succeeded_run=("success_int", "sum"),
    )
    per_run["success_rate"] = per_run["succeeded_run"] / per_run["attempted_run"]
    avg = per_run.groupby(["agent_id", "model_name", "job_difficulty"], as_index=False).agg(
        avg_success_rate_per_run=("success_rate", "mean"),
        std_success_rate_per_run=("success_rate", "std"),
        n_runs_with_attempt=("run_id", "nunique"),
    )
    return pooled.merge(avg, on=["agent_id", "model_name", "job_difficulty"], how="left")


def metric_energy_spent_per_phase(memory_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty:
        return pd.DataFrame()
    cols = ["energy_vote", "energy_decide", "energy_act", "total_energy_spent"]
    per_run = memory_df.groupby(["agent_id", "model_name", "run_id"])[cols].mean().reset_index()
    out = per_run.groupby(["agent_id", "model_name"])[cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def metric_category_difficulty_attempts(memory_df: pd.DataFrame) -> pd.DataFrame:
    attempts = _attempts(memory_df)
    if attempts.empty:
        return pd.DataFrame()
    attempts = attempts[attempts["job_category"].notna() & attempts["job_difficulty"].notna()].copy()
    keys = ["agent_id", "model_name", "run_id"]
    run_counts = attempts.groupby(keys + ["job_category", "job_difficulty"]).size().reset_index(name="count_per_run")
    pooled = run_counts.groupby(["agent_id", "model_name", "job_category", "job_difficulty"], as_index=False).agg(
        total_attempts=("count_per_run", "sum"),
        avg_count_per_run=("count_per_run", "mean"),
        std_count_per_run=("count_per_run", "std"),
        n_runs=("run_id", "nunique"),
    )
    return pooled


def metric_energy_over_time(memory_df: pd.DataFrame, agent_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty or agent_df.empty:
        return pd.DataFrame()
    rows = []
    for _, arow in agent_df.iterrows():
        run_id = arow["run_id"]
        agent_id = arow["agent_id"]
        max_rounds = arow.get("max_rounds")
        if pd.isna(max_rounds):
            max_rounds = int(memory_df[memory_df["run_id"] == run_id]["round_num"].max())
        sub_mem = memory_df[(memory_df["run_id"] == run_id) & (memory_df["agent_id"] == agent_id)]
        energy_by_round = dict(zip(sub_mem["round_num"], sub_mem["energy_start"]))
        for round_num in range(1, int(max_rounds) + 1):
            rows.append({
                "agent_id": agent_id,
                "model_name": arow.get("model_name"),
                "run_id": run_id,
                "round_num": round_num,
                "energy": float(energy_by_round.get(round_num, 0.0)),
                "point_type": "round_start",
            })
        rows.append({
            "agent_id": agent_id,
            "model_name": arow.get("model_name"),
            "run_id": run_id,
            "round_num": int(max_rounds) + 1,
            "energy": float(arow.get("final_energy", 0.0) or 0.0),
            "point_type": "final",
        })
    long_df = pd.DataFrame(rows)
    return long_df.groupby(["agent_id", "model_name", "round_num", "point_type"], as_index=False).agg(
        avg_energy=("energy", "mean"),
        std_energy=("energy", "std"),
        min_energy=("energy", "min"),
        max_energy=("energy", "max"),
        n_runs=("run_id", "nunique"),
    )


def aggregate_experiment(experiment_dir: str | Path) -> dict[str, pd.DataFrame]:
    memory_df, agent_df, run_df = load_experiment_runs(experiment_dir)
    vote_df = load_experiment_recommendations(experiment_dir)
    return {
        "runs": run_df,
        "recommendations_raw": vote_df,
        "recommended_job_collisions": metric_recommended_job_collisions(vote_df),
        "recommended_other_only_job_collisions": metric_recommended_other_only_job_collisions(vote_df),
        "recommended_donations_to_self": metric_recommended_donations_to_self(vote_df),
        "recommendation_action_self_vs_others": metric_recommendation_action_self_vs_others(vote_df),
        "recommendation_follow_rate": metric_recommendation_follow_rate(vote_df, memory_df),
        "recommendation_decision_follow_source": metric_recommendation_decision_follow_source(vote_df, memory_df),
        "enacted_recommendation_net_energy": metric_enacted_recommendation_net_energy(vote_df, memory_df),
        "average_collisions": metric_average_collisions(memory_df),
        "energy_gained_per_spent": metric_energy_gained_per_spent(memory_df),
        "rounds_active": metric_rounds_active(agent_df),
        "first_deactivation": metric_first_deactivation(agent_df),
        "donations_received": metric_donations_received(memory_df),
        "donations_made": metric_donations_made(memory_df),
        "action_frequency": metric_action_frequency(memory_df),
        "action_frequency_with_errors": metric_action_frequency_with_errors(memory_df),
        "attempted_difficulty_distribution": metric_attempted_difficulty_distribution(memory_df),
        "success_rate_by_difficulty": metric_success_rate_by_difficulty(memory_df),
        "energy_spent_per_phase": metric_energy_spent_per_phase(memory_df),
        "category_difficulty_attempts": metric_category_difficulty_attempts(memory_df),
        "energy_over_time": metric_energy_over_time(memory_df, agent_df),
    }


def export_aggregate_metrics(metrics: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in metrics.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path}")


def main():
    run_dir = Path(__file__).resolve().parents[1] / "results" / "coop" / "exponent_0.5_sabotage"
    out_dir = run_dir / "aggregate"
    export_aggregate_metrics(aggregate_experiment(run_dir), out_dir)


if __name__ == "__main__":
    main()

