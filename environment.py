from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, Send
from langgraph.runtime import Runtime
from memory import MemoryEntry
from jobs import Difficulty, Job
import operator
from jobs import JobGenerator
from agent_middleware import AgentVote, AgentVoteWithSabotage, EnergyAgent, IDLE_ENERGY_COST, SABOTAGE_COST
from collections import defaultdict


@dataclass
class ExperimentConfig:
    """Class for choosing which experiment is run"""
    name: str
    prompt_mode: Literal["comp", "coop"]
    use_votes: bool
    enable_sabotage: bool = False
    
class VoteResult(TypedDict):
    """What an agent voted in the discussion phase."""
    agent_id: str
    vote: AgentVote | AgentVoteWithSabotage
    energy_before: float
    energy_after: float

class Decision(TypedDict):
    """The decision the agent makes each round in agent_decide()."""
    agent_id: str               # ID of this decision's agent
    action: str                 # The chosen action
    job_id: str | None          # The chosen job if action=ATTEMPT_JOB
    target_agent: str | None    # The agent to donate to if action=DONATE
    donate_amount: float | None # The donated amount if action=DONATE
    energy_before: float        # Before decide() was called
    energy_after: float         # After decide() was called
    

class JobAttemptResult(TypedDict):
    """The result of one agent's job attempt in agent_attempt_job() each round."""
    agent_id: str         # ID of this decision's agent
    job_id: str | None    # The chosen job if action=ATTEMPT_JOB
    succeeded: bool       # Whether the attempt was successful
    energy_before: float  # Before attempt_job() was called
    energy_after: float   # After attempt_job() was called
    agent_answer: str     # The agent's answer


# The state of the environment. Passed between graph nodes and ultimately returned by the graph.
class EnvironmentState(TypedDict):
    """The state of the environment. Passed between graph nodes and ultimately returned by the graph."""
    max_rounds: int # Simulation will run for this number of rounds or until all agents deactivate
    round_num: int  # Current round
    prompt_mode: Literal["comp", "coop"] # prompt mode
    use_votes: bool
    enable_sabotage: bool
    jobs: list[Job] # Availble jobs for the current round
    decisions: Annotated[list[Decision], operator.add] # List of all agent decisions. Using reducer function operator.add to append agent decisions instead of overwriting.
    agents: list[EnergyAgent] # All agents in the environment
    job_attempt_results: Annotated[list[JobAttemptResult], operator.add] # List of all job attempt results.
    vote_results: Annotated[list[VoteResult], operator.add]
    sabotaged_jobs: list[str] # List of all sabotaged jobs


@dataclass
class SimulationContext:
    """Context schema for runtime context passed to start_round node (info that is not part of state)."""
    job_generator: JobGenerator
    num_jobs: int
    difficulty_distribution: dict[Difficulty, int]
    
    
class DecideState(TypedDict):
    """
    The state for all the spawned agent_decide nodes.
    Contains:
     - agent: EnergyAgent (the agent deciding)
     - round_num: int (the current round)
     - jobs: list[Job] (all jobs in this round)
     - other_agents: list[EnergyAgent] (all other agents)
     - votes: list[AgentVote] (all votes from discussion phase)
     - prompt_mode: str (whether it is competitive or cooperative experiment)
    """
    agent: EnergyAgent
    round_num: int
    jobs: list[Job]
    other_agents: list[EnergyAgent]
    votes: list[AgentVote | AgentVoteWithSabotage] # Empty when there is no discussion phase
    prompt_mode: Literal["comp", "coop"] # prompt mode
    
    
class VoteState(TypedDict):
    """State for all spawned vote nodes"""
    agent: EnergyAgent
    round_num: int
    jobs: list[Job]
    all_agents: list[EnergyAgent]    
    prompt_mode: Literal["comp", "coop"]

class AttemptState(TypedDict):
    """
    The state for all the spawned agent_attempt_job nodes.
    Contains:
     - agent: EnergyAgent (the agent attempting the job)
     - round_num: int (the current round)
     - job: Job (the attempted job)
    """
    agent: EnergyAgent
    job: Job
    round_num: int


def start_round(state: EnvironmentState, runtime: Runtime[SimulationContext]) -> dict[str, Any]:
    """
    The first node to run in each round:
     - Generates the jobs for the round via JobGenerator.
     - Resets list of agent decisions.
     - Resets dict of how jobs are assigned
     - Resets stored votes

    Returns:
        dict[str, Any]: Updates the 'round_num', 'jobs' and 'decisions' fields in the EnvironmentState.
    """
    job_generator = runtime.context.job_generator
    
    jobs = job_generator.generate_jobs(runtime.context.num_jobs, runtime.context.difficulty_distribution)
    
    for agent in state["agents"]:
        agent.energy_at_round_start = float(agent.energy)
    
    # Printing information
    active_agents = [agent for agent in state["agents"] if agent.is_active]
    print(f"\n{'-'*50}")
    print(f"ROUND {state['round_num']}")
    print(f"Active agents this round {[a.agent_id for a in active_agents]}")
    print(f"{'-'*50}")
    
    return {
        "jobs": jobs,
        "decisions": Overwrite([]), # Overwrite bypasses the reducer and directly overwrites the state value.
        "assigned_jobs": {},
        "job_attempt_results": Overwrite([]),
        "vote_results": Overwrite([]),
        "sabotaged_jobs": Overwrite([]),
    }


def start_decisions(state: EnvironmentState) -> list[Send]:
    """Conditional edge function to spawn agent_decide nodes for each active agent every round."""
    active_agents = [agent for agent in state["agents"] if agent.is_active]
    
    # For every active agent spawn an agent_decide node with the DecideState fields filled.
    return [Send("agent_decide", {"agent": a, 
                                  "round_num": state["round_num"], 
                                  "jobs": state["jobs"], 
                                  "other_agents": [agent for agent in state["agents"] if agent.agent_id != a.agent_id],
                                  "votes": None,
                                  "prompt_mode": state["prompt_mode"],
                                  }
                 ) for a in active_agents]


def agent_decide(state: DecideState) -> dict[str, list[Decision]]:
    """
    Node that asks an agent to decide what to do in the round.
    Runs in parallel for each active agent in the environment.

    Returns:
        dict[str, list[Decision]]: Updates the 'decisions' field in EnvironmentState by returning 
        a list of a single Decision object that is appended to the list of all decisions by the reducer function.
    """
    # Get the agent (passed to state by conditional edge start_decisions)
    agent = state["agent"]
    
    # Call LLM to get agents decision
    energy_before = agent.energy
    print(f"{agent.agent_id} energy before deciding: {energy_before}")
    decision = agent.decide(round_num=state["round_num"], 
                            jobs=state["jobs"], 
                            other_agents=state["other_agents"], 
                            prompt_mode=state["prompt_mode"], 
                            votes=state.get("votes")
                            )
    energy_after = agent.energy
    print(f"{agent.agent_id} energy after deciding: {energy_after}")
    
    # Add "job_" infront of response if it isn't already there -> fixes number only returns.
    job_id = decision.job_id
    if job_id is not None and not job_id.startswith("job_"):
        job_id = f"job_{job_id}"
        
    # Return agents decision. Is appended to the list of decisions in EnvironmentState by the reducer function.
    return {"decisions": [Decision(
        agent_id=agent.agent_id,
        action=decision.action,
        job_id=job_id,
        target_agent=decision.target_agent,
        donate_amount=decision.amount,
        energy_before=energy_before,
        energy_after=energy_after,
    )]}


def idle_and_donate(state: EnvironmentState):
    """
    Handles idle and donation (and sabotage) actions before job actions are done.
    These actions don't need parallel execution since they have no LLM call.
    Also donating in parallel would likely require race conditions to be handled.
    """
    # Dict that maps agent ids to agents {agent_id: agent}
    agents_by_id = {a.agent_id: a for a in state["agents"]}
    sabotaged_jobs = []
    
    for decision in state["decisions"]:
        # Get the id of the agent from the decision and get agent from dict
        agent = agents_by_id[decision["agent_id"]]
        
        # Skip deactivated agents
        if not agent.is_active:
            continue
        
        # Handle idle actions
        if decision["action"] == "IDLE":
            agent.idle()
        
        # Handle donate actions
        elif decision["action"] == "DONATE":
            # Get the target agent from the dict via the id in decision
            target = agents_by_id.get(decision["target_agent"])
            if target is None:
                print(f"ERROR: {agent.agent_id} tried to donate to unknown agent {decision["target_agent"]}, skipping.")
                continue
            agent.donate_energy(target, decision["donate_amount"])
            
        elif decision["action"] == "SABOTAGE_JOB":
            sabotaged_jobs.append(decision["job_id"])
            agent.energy = max(0.0, agent.energy - SABOTAGE_COST)
    
    # For experiments without sabotage this just always returns an empty list
    return {"sabotaged_jobs": sabotaged_jobs}
        
        
def start_job_attemtps(state: EnvironmentState) -> list[Send]:
    """Conditional edge function to spawn agent_decide nodes for each active agent every round."""
    # Dicts mapping ids to objects in the state {job_id: job} {agent_id: agent}
    jobs_by_id = {j.job_id: j for j in state["jobs"]}
    agents_by_id = {a.agent_id: a for a in state["agents"]}
    
    # Spawn agent_attempt_job node for all active agents who chose ATTEMPT_JOB:
    sends = [Send("agent_attempt_job", {"agent": agents_by_id[dec["agent_id"]],
                                       "job": jobs_by_id[dec["job_id"]],
                                       "round_num": state["round_num"],
                                       }
                 ) for dec in state["decisions"] if dec["action"] == "ATTEMPT_JOB" and agents_by_id[dec["agent_id"]].is_active and dec["job_id"] in jobs_by_id]
    
    if not sends:
        return "reward_succeeded_jobs_and_update_memory" # Skip attempt job phase if no one decided to attempt a job
    else:
        return sends


def agent_attempt_job(state: AttemptState) -> dict[str, list[JobAttemptResult]]:
    """
    Node that asks one agent to attempt their chosen job.
    Runs in parallel for each agent.
    
    Returns:
    dict[str, list[JobAttemptResult]]: Updates the 'job_attempt_results' list in EnvironmentState by returning 
    a list of a single JobAttemptResult object that is appended to the list in environment by the reducer function.
    """
    agent = state["agent"]
    job = state["job"]
    
    energy_before = agent.energy
    print(f"{agent.agent_id} energy before attempting job: {energy_before}")
    result = agent.attempt_job(job=job, round_num=state["round_num"])
    energy_after = agent.energy
    print(f"{agent.agent_id} energy after attempting job: {energy_after}")
    
    succeeded = job.evaluate(result.answer) # check if answer was correct
    
    return {"job_attempt_results": [JobAttemptResult(
        agent_id=agent.agent_id,
        job_id=job.job_id,
        succeeded=succeeded,
        energy_before=energy_before,
        energy_after=energy_after,
        agent_answer=result.answer,
    )]}
    
    
def reward_succeeded_jobs_and_update_memory(state: EnvironmentState):
    """
    Splits the reward of all jobs between all agents who successfully completed the job.
    Then updates the memory of all agents.
    Both are done in this node for efficiency, since reward calculations can be used 
    directly in memory update.
    """
    # Dicts mapping ids to objects in the state {job_id: job} {agent_id: agent}
    jobs_by_id = {j.job_id: j for j in state["jobs"]}
    agents_by_id = {a.agent_id: a for a in state["agents"]}

    # Dict mapping job_id to all agents who succeeded
    successful_agents_by_job = defaultdict(list)
    # for all attempts, if they succeeded, store the agents id in the list of successful agents for that job id.
    for attempt in state["job_attempt_results"]:
        if attempt["succeeded"]:
            # Adding the agent_id to the values list of the job_id key in the dict
            print(f"{attempt['agent_id']} SUCCEEDED solving job {attempt['job_id']}")
            successful_agents_by_job[attempt["job_id"]].append(attempt["agent_id"])
        else:
            print(f"{attempt['agent_id']} FAILED solving job {attempt['job_id']}")
    
    # Go through all successful agents and split the job reward among them:
    reward_by_agent = {}
    for job_id, successful_agents in successful_agents_by_job.items():
        job = jobs_by_id[job_id]
        if job_id in state["sabotaged_jobs"]:
            split_reward = 0
        else:
            split_reward = round(job.reward / len(successful_agents), 2)
        for agent_id in successful_agents:
            # Reward the agent with their split
            print(f"{agent_id} was rewarded {split_reward} energy")
            agents_by_id[agent_id].add_energy(split_reward)
            reward_by_agent[agent_id] = split_reward


    # ---- Build and store a MemoryEntry for every agent that made a decision ----
    # Lookup dict to quickly find the attempt from agent_id without scanning through dict.
    attempt_by_agent = {attempt["agent_id"]: attempt for attempt in state["job_attempt_results"]}
    # Lookup dict to quickly find VoteResult from agent_id
    vote_result_by_agent = {vr["agent_id"]: vr for vr in state["vote_results"]}
    
    # Lookup of donations received this round: {agent_id: [{"from_agent": ..., "amount": ...}]}
    received_by_agent: dict[str, list[dict]] = defaultdict(list)
    for decision in state["decisions"]:
        if decision["action"] == "DONATE" and decision["target_agent"]:
            if decision["target_agent"] not in agents_by_id:
                continue # Agent tried to donate to invalid target
            # decision["energy_after"] is energy after decision was made. Can't donate more than they have.
            actual_donated = min(decision["donate_amount"], decision["energy_after"])
            received_by_agent[decision["target_agent"]].append({
                "from_agent": decision["agent_id"],
                "amount": round(actual_donated, 2),
            })
    
    # Build memory entry
    for decision in state["decisions"]:
        agent = agents_by_id[decision["agent_id"]]
        entry = _build_memory_entry(
            decision=decision,
            round_num = state["round_num"],
            jobs_by_id=jobs_by_id,
            attempt_by_agent=attempt_by_agent,
            reward_by_agent=reward_by_agent,
            agents_by_id=agents_by_id,
            vote_result_by_agent=vote_result_by_agent,
            received_by_agent=received_by_agent,
        )
        # Unknown action (should be unreachable)
        if entry is None:
            continue
        
        agent.memory.append(entry)
        
    # Write a memory entry for reactivated agents who received a donation but made no decision this round (they were inactive when decisions where made)
    agents_with_decisions = {dec["agent_id"] for dec in state["decisions"]}
    for agent_id, donations in received_by_agent.items():
        if agent_id in agents_with_decisions:
            continue # already handled above
        agent = agents_by_id[agent_id]
        agent.memory.append(MemoryEntry(
            round_num=state["round_num"],
            action="INACTIVE",
            energy_spent_deciding=0.0,
            energy_spent_on_action=0.0,
            energy_spent_voting=None,
            energy_at_start_of_round=agent.energy_at_round_start,
            received_donations=donations,
        ))


def _build_memory_entry(decision: Decision, 
                        round_num: int, jobs_by_id: dict[str, Job],
                        attempt_by_agent: dict[str, JobAttemptResult], 
                        reward_by_agent: dict[str, float], 
                        agents_by_id: dict[str, EnergyAgent],
                        vote_result_by_agent: dict[str, VoteResult],
                        received_by_agent: dict[str, list[dict]],
                        ) -> MemoryEntry | None:
    """
    Helper for buidling the memory entry based on the action taken.
    """
    action = decision["action"]
    energy_spent_deciding = round(decision["energy_before"] - decision["energy_after"], 2)
    agent_id = decision["agent_id"]
    energy_at_start_of_round = agents_by_id[agent_id].energy_at_round_start
    vote_result = vote_result_by_agent.get(agent_id)
    energy_spent_on_vote = round(vote_result["energy_before"] - vote_result["energy_after"], 2) if vote_result else None
    received_donations = received_by_agent.get(agent_id)
    
    attempts_by_job: dict[str, list[str]] = defaultdict(list)
    for attempt in attempt_by_agent.values():
        if attempt["job_id"]:
            attempts_by_job[attempt["job_id"]].append(attempt["agent_id"])
    
    match action:
        case "ATTEMPT_JOB":
            attempt = attempt_by_agent.get(agent_id)
            job = jobs_by_id.get(decision["job_id"])
            
            if attempt is None:
                # Agent ran out of energy while deciding and thus never got to attempt the job.
                return MemoryEntry(
                    round_num=round_num,
                    action=action,
                    energy_spent_deciding=energy_spent_deciding,
                    energy_spent_on_action=0.0,
                    succeeded=None, # None = deactivated before attempt
                    attempted_job=job,
                    energy_at_start_of_round=energy_at_start_of_round,
                    energy_spent_voting=energy_spent_on_vote,
                    received_donations=received_donations,
                )
            else:
                succeeded = attempt["succeeded"] # whether agent solved the job correctly
                return MemoryEntry(
                    round_num=round_num,
                    action=action,
                    energy_spent_deciding=energy_spent_deciding,
                    energy_spent_on_action=round(attempt["energy_before"] - attempt["energy_after"], 2),
                    succeeded=succeeded,
                    energy_rewarded=reward_by_agent[agent_id] if succeeded else 0,
                    energy_at_start_of_round=energy_at_start_of_round,
                    attempted_job=job,
                    agent_answer=attempt["agent_answer"],
                    energy_spent_voting=energy_spent_on_vote,
                    received_donations=received_donations,
                )
        
        case "IDLE":
            return MemoryEntry(
                round_num=round_num,
                action=action,
                energy_spent_deciding=energy_spent_deciding,
                energy_spent_on_action=IDLE_ENERGY_COST,
                energy_at_start_of_round=energy_at_start_of_round,
                energy_spent_voting=energy_spent_on_vote,
                received_donations=received_donations,
            )
    
        case "DONATE":
            # decision["energy_after"] is the energy remaining after deciding, which caps the donation.
            actual_donated = min(decision["donate_amount"], decision["energy_after"])
            rounded_donation = round(actual_donated, 2)
            return MemoryEntry(
                round_num=round_num,
                action=action,
                energy_spent_deciding=energy_spent_deciding,
                energy_spent_on_action=rounded_donation,
                target_agent=decision["target_agent"],
                donated_amount=rounded_donation,
                energy_at_start_of_round=energy_at_start_of_round,
                energy_spent_voting=energy_spent_on_vote,
                received_donations=received_donations,
            )
        
        case "SABOTAGE_JOB":
            job = jobs_by_id.get(decision["job_id"])
            hit_agents = attempts_by_job.get(decision["job_id"], [])
            return MemoryEntry(
                round_num=round_num,
                action=action,
                energy_spent_deciding=energy_spent_deciding,
                energy_spent_on_action=SABOTAGE_COST,
                energy_spent_voting=energy_spent_on_vote,
                energy_at_start_of_round=energy_at_start_of_round,
                received_donations=received_donations,
                sabotaged_job=job,
                sabotage_successful=len(hit_agents) > 0,
                sabotage_hit_agents=hit_agents,
            )
        
        case _:
            return None
    


def end_round(state: EnvironmentState) -> dict[str, int]:
    """Logs deactivation and increments the round number."""
    # Logging deactivation:
    for agent in state["agents"]:
        if not agent.is_active and agent.deactivated_at_round is None:
            # This is the round of the agent's first deactivation
            agent.deactivated_at_round = state["round_num"]
            print(f"{agent.agent_id} DEACTIVATED at round {state['round_num']}")
            
        if agent.is_active:
            agent.rounds_active += 1
        
    # Incrementing round num:
    return {"round_num": state["round_num"] + 1}


def should_continue(state: EnvironmentState) -> str:
    """Conditional edge checking if simulation has ended. Returns the name of the next node."""
    all_inactive = all(not a.is_active for a in state["agents"])
    if all_inactive:
        print("\n Simulation end: All agents deactivated!")
        return END
    elif state["round_num"] > state["max_rounds"]:
        print(f"Simulation end: Reached max rounds ({state['max_rounds']})")
        return END
    else:
        return "start_round"




# --------------------- Vote nodes / edges --> for the discussion phase ---------------------

def start_votes(state: EnvironmentState) -> list[Send]:
    """Conditional edge function to spawn agent_vote nodes for each active agent every round."""
    active_agents = [a for a in state["agents"] if a.is_active]
    print(f"Active agents voting this round {[a.agent_id for a in active_agents]}")
    
    # Skip voting if only 1 agent remains
    if len(active_agents) <= 1:
        return [Send("agent_decide", {
            "agent": a,
            "round_num": state["round_num"],
            "jobs": state["jobs"],
            "other_agents": [ag for ag in state["agents"] if ag.agent_id != a.agent_id],
            "votes": [],
            "prompt_mode": state["prompt_mode"],
        }) for a in active_agents]
        
    return [Send("agent_vote", {"agent": a,
                                "round_num": state["round_num"],
                                "jobs": state["jobs"],
                                "all_agents": state["agents"],
                                "prompt_mode": state["prompt_mode"],
                                }
                 ) for a in active_agents]


def agent_vote(state: VoteState) -> dict[str, list[AgentVote]]:
    """
    Node that asks an agent to make a vote (in the discussion phase).
    Runs in parallel for each active agent in the environment.

    Returns:
        dict[str, list[AgentVote]]: Updates the 'vote_results' field in EnvironmentState by returning 
        a list of a single AgentVote object that is appended to the list of all votes by the reducer function.
    """
    agent = state["agent"]
    print(f"{agent.agent_id} casting vote...")
    
    energy_before = agent.energy
    vote = agent.vote(
        round_num=state["round_num"],
        jobs=state["jobs"],
        all_agents=state["all_agents"],
        prompt_mode = state["prompt_mode"]
    )
    energy_after = agent.energy
    
    # Reducer will add to list of VoteResults
    return {"vote_results": [VoteResult(agent_id=agent.agent_id, vote=vote, energy_before=energy_before, energy_after=energy_after)]}


def collect_votes(state: EnvironmentState):
    """Node we simply pass through to allow for a fan-in after the vote fan-out."""
    pass


def start_decisions_with_votes(state: EnvironmentState) -> list[Send]:
    """Conditional edge function to spawn agent_decide nodes with the votes from discussion phase for each active agent every round."""
    active_agents = [agent for agent in state["agents"] if agent.is_active]
    if not active_agents:
        return "idle_and_donate" # skip decide, no agents active -> nothing to do.
    # For every active agent spawn an agent_decide node with the DecideState fields filled.
    return [Send("agent_decide", {"agent": a, 
                                  "round_num": state["round_num"], 
                                  "jobs": state["jobs"], 
                                  "other_agents": [agent for agent in state["agents"] if agent.agent_id != a.agent_id],
                                  "votes": [vr["vote"] for vr in state["vote_results"] if vr["vote"].recommendations], # Skip if no recommendations
                                  "prompt_mode": state["prompt_mode"],
                                  }
                 ) for a in active_agents]



def add_shared_nodes(builder: StateGraph):
    """Adds nodes to the simulation graph that experiments share."""
    builder.add_node("start_round", start_round)
    builder.add_node("agent_decide", agent_decide)
    builder.add_node("idle_and_donate", idle_and_donate)
    builder.add_node("agent_attempt_job", agent_attempt_job)
    builder.add_node("reward_succeeded_jobs_and_update_memory", reward_succeeded_jobs_and_update_memory)
    builder.add_node("end_round", end_round)


def add_shared_edges(builder: StateGraph):
    """Adds edges to the simulation graph that experiments share."""
    builder.add_edge(START, "start_round")
    builder.add_edge("agent_decide", "idle_and_donate")
    builder.add_conditional_edges("idle_and_donate", start_job_attemtps, ["agent_attempt_job", "reward_succeeded_jobs_and_update_memory"])
    builder.add_edge("agent_attempt_job", "reward_succeeded_jobs_and_update_memory")
    builder.add_edge("reward_succeeded_jobs_and_update_memory", "end_round")
    builder.add_conditional_edges("end_round", should_continue)


def build_simulation_graph(use_votes: bool):
    builder = StateGraph(EnvironmentState)
    
    # Nodes
    add_shared_nodes(builder)
    if use_votes:
        builder.add_node("agent_vote", agent_vote)
        builder.add_node("collect_votes", collect_votes)
    
    # Edges
    add_shared_edges(builder)
    if use_votes:
        builder.add_conditional_edges("start_round", start_votes, ["agent_vote", "agent_decide"])
        builder.add_edge("agent_vote", "collect_votes")
        builder.add_conditional_edges("collect_votes", start_decisions_with_votes, ["agent_decide", "idle_and_donate"])
    else:
        builder.add_conditional_edges("start_round", start_decisions, ["agent_decide"])
        
    # Compile graph
    return builder.compile()
    

def run_simulation(max_rounds: int, 
                   agents: list[EnergyAgent], 
                   job_generator: JobGenerator, 
                   num_jobs: int, 
                   difficulty_distribution: dict[Difficulty, int],
                   experiment_config: ExperimentConfig,
                   ) -> EnvironmentState:
    """
    Runs the simulation.
    
    Args:
        max_rounds (int): The number of rounds the simulation will run for if it doesn't stop early due to all agents deactivating.
        agents (list[EnergyAgent]): All the agents in the environment.
        job_generator (JobGenerator): An instance of the JobGenerator class.
        num_jobs (int): The number of jobs in each round.
        difficulty_distribution (dict[Difficulty, int]): The difficulty distribution of jobs in the environment.
        experiment_config (ExperimentConfig): Config of the experiment to run.

    Returns:
        EnvironmentState: The final EnvironmentState after the simulation ends.
    """
    # Build the graph and setup initial state and context.
    graph = build_simulation_graph(use_votes=experiment_config.use_votes)

    init_state = EnvironmentState(max_rounds=max_rounds, 
                                  round_num=1, 
                                  agents=agents,
                                  prompt_mode=experiment_config.prompt_mode,
                                  use_votes=experiment_config.use_votes,
                                  enable_sabotage=experiment_config.enable_sabotage,
                                  )
    
    context = SimulationContext(job_generator=job_generator, 
                                num_jobs=num_jobs, 
                                difficulty_distribution=difficulty_distribution,
                                )
    
    # Run the simulation by invoking the graph
    return graph.invoke(init_state, context=context)
    