from dataclasses import dataclass, field
from jobs import Job


# OBS: This class ended up both storing information agent sees
# and information purely used for logging and analysing results in the end.
@dataclass
class MemoryEntry:
    """
    Records the outcome of each round.
    Stored as a list of entries in EnergyAgent.
    Agents have access to memory from the last 10 rounds in the prompt.
    """
    round_num: int
    action: str
    energy_spent_deciding: float
    energy_spent_on_action: float
    energy_spent_voting: float | None
    
    # ATTEMPT_JOB fields:
    attempted_job: Job | None = None
    succeeded: bool | None = None
    energy_rewarded: float | None = None
    
    # DONATE fields:
    target_agent: str | None = None
    donated_amount: float | None = None
    
    # SABOTAGE_JOB fields:
    sabotaged_job: Job | None = None
    sabotage_successful: bool | None = None
    sabotage_hit_agents: list[str] | None = None
    
    # Other fields for metrics / logging
    energy_at_start_of_round: float = 0.0
    agent_answer: str | None = None
    received_donations: list[dict] | None = None # [{"from_agent": "agent_1", "amount": 400.0}]