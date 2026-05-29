# --------------------------------------------------------------------------------------------
#This file is used to save simulation results to JSON.

# It creates four files at output_dir:
# - agents.json --> Full per-agent data including memory entries
# - config.json --> The run configuration passed in
# - transcript.json --> Full log of reasoning content and output of every agent
# - token_usage.json --> Token usage for each agent for each phase of every round
# --------------------------------------------------------------------------------------------

import json
import os
from typing import Any

from agent_middleware import EnergyAgent, ErrorEntry, TranscriptEntry, TokenEntry
from jobs import Job
from memory import MemoryEntry


# --------------------- Helpers to serialize jobs, memory and agent info ---------------------

def _serialise_job(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "category": job.category,
        "difficulty": job.difficulty.string,
        "reward": job.reward,
        "description": job.description,
        "options": job.options,
        "correct_answer": job.answer, 
    }

def _serialise_memory_entry(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "round_num": entry.round_num,
        "action": entry.action,
        "energy_spent_voting": entry.energy_spent_voting,
        "energy_spent_deciding": entry.energy_spent_deciding,
        "energy_spent_on_action": entry.energy_spent_on_action,
        "energy_at_start_of_round": entry.energy_at_start_of_round,
        
        # Fields for if action was ATTEMPT_JOB:
        "attempted_job": _serialise_job(entry.attempted_job) if entry.attempted_job else None,
        "succeeded": entry.succeeded,
        "energy_rewarded": entry.energy_rewarded,
        "agent_answer": entry.agent_answer,
        
        # Fields for if aciton was DONATE:
        "target_agent": entry.target_agent,
        "donated_amount": entry.donated_amount,
        "received_donations": entry.received_donations,
        
        # Fields for if action was SABOTAGE_JOB:
        "sabotaged_job": _serialise_job(entry.sabotaged_job) if entry.sabotaged_job else None,
        "sabotage_successful": entry.sabotage_successful,
        "sabotage_hit_agents": entry.sabotage_hit_agents,
    }

def _serialise_error_entry(entry: ErrorEntry) -> dict[str, Any]:
    return {
        "round_num": entry.round_num,
        "phase": entry.phase,
        "error_type": entry.error_type,
        "error_message": entry.error_message,
    }

def _serialise_agent(agent: EnergyAgent) -> dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "model_name": agent.model_name,
        "model_size": agent.model_size,
        "init_energy": agent.init_energy,
        "final_energy": agent.energy,
        "rounds_active": agent.rounds_active,
        "deactivated_at_round": agent.deactivated_at_round,
        "memory": [_serialise_memory_entry(e) for e in agent.memory],
        "errors": [_serialise_error_entry(e) for e in agent.error_log],
    }

def _serialise_transcript_entry(entry: TranscriptEntry) -> dict[str, Any]:
    return {
        "round_num": entry.round_num,
        "phase": entry.phase,
        "agent_id": entry.agent_id,
        "reasoning_content": entry.reasoning_content,
        "structured_response": entry.structured_response,
    }
    
def _serialise_token_entry(entry: TokenEntry) -> dict[str, Any]:
    return {
        "round_num": entry.round_num,
        "phase": entry.phase,
        "agent_id": entry.agent_id,
        "output_tokens": entry.output_tokens,
    }




# --------------------- Functions for creating all the files ---------------------

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")


def save_agents(agents: list[EnergyAgent], output_dir: str):
    """Creates agents.json"""
    agents_path = os.path.join(output_dir, "agents.json")
    _write_json(agents_path, [_serialise_agent(a) for a in agents])


def save_config(config: dict[str, Any], output_dir: str):
    """Creates config.json"""
    config_path = os.path.join(output_dir, "config.json")
    _write_json(config_path, config)


def save_transcript(agents: list[EnergyAgent], output_dir: str):
    """Creates transcript.json"""
    phase_order = {"vote": 0, "decide": 1, "act": 2}
    
    all_entries = []
    for agent in agents:
        all_entries.extend(agent.transcript_log)
        # Equivalent to:
        # for entry in agent.transcript_log:
        #     all_entries.append(entry)
    
    # Sort each entry based on the key. 
    # Here the key is a tuple of the entry's round_num, phase number via phase_order and agent_id.
    # So it sorts first by round number, then by phase and then by agent id.
    all_entries.sort(key=lambda e: (e.round_num, phase_order.get(e.phase), e.agent_id))
    
    transcript_path = os.path.join(output_dir, "transcript.json")
    _write_json(transcript_path, [_serialise_transcript_entry(e) for e in all_entries])


def save_token_usage(agents: list[EnergyAgent], output_dir: str):
    """Creates token_usage.json"""
    phase_order = {"vote": 0, "decide": 1, "act": 2}
    
    all_entries = []
    for agent in agents:
        all_entries.extend(agent.token_log)
        
    all_entries.sort(key=lambda e: (e.round_num, phase_order.get(e.phase), e.agent_id))
    
    token_usage_path = os.path.join(output_dir, "token_usage.json")
    _write_json(token_usage_path, [_serialise_token_entry(e) for e in all_entries])


def save_results(agents: list[EnergyAgent], config: dict[str, Any], output_dir: str):
    """
    Saves agent data and run config as JSON files into output_dir.
    
    Args:
        agents (list[EnergyAgent]): The list of EnergyAgent objects AFTER the simulation ends.
        config (dict[str, Any]): A dict describing the run. E.g. keys could include name/type of experiment, max_rounds, num_jobs, difficulty_distribution.
        output_dir (str): Path to write the files into. Will be created if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True) # exist_ok=True to avoid raising error if directory alredy exists.
    save_agents(agents, output_dir)
    save_config(config, output_dir)
    save_transcript(agents, output_dir)
    save_token_usage(agents, output_dir)
