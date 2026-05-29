from dataclasses import dataclass
import operator
from typing import Annotated, Any, Literal
from typing_extensions import Self
from langchain.messages import AIMessage, AnyMessage
from langgraph.graph import add_messages
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents import AgentState, create_agent
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, model_validator
from prompts import build_decision_prompt, build_job_solve_prompt, build_vote_prompt
from jobs import Job
from memory import MemoryEntry


# The flat cost of choosing to idle
IDLE_ENERGY_COST = 10
SABOTAGE_COST = 10

# Cost per token = ENERGY_COST_SCALE * model_size**MODEL_SIZE_EXPONENT
ENERGY_COST_SCALE = 0.015
MODEL_SIZE_EXPONENT = 0.5



class EnergyState(AgentState):
    """
    Custom state that stores the agents energy during a LLM call run.
    LLM call is graph that might have multiple calls and this stores the energy across those calls.
    When all the steps are done and the LLM call is completely finished the actual energy
    field of the EnergyAgent is updated accordingly.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    # This stores the energy during an LLM call. The EnergyAgent class holds energy between calls.
    energy: float
    init_energy: float
    phase_log: Annotated[list[dict], operator.add] # Stores reasoning_content and output_tokens for each phase
            
 
        
# Using Class-based middleware to combine multiple hooks in a single middleware.
# When doing this class needs to inherit from AgentMiddleware
# https://docs.langchain.com/oss/python/langchain/middleware/custom

# The custom state is defined via middleware since according to docs this is the preferred way.
# https://docs.langchain.com/oss/python/langchain/agents#defining-state-via-middleware
class EnergyMiddleware(AgentMiddleware):
    """
    Captures the number of tokens generated after each model call
    and deducts energy from the EnergyState based on this.
    """
    state_schema = EnergyState
    
    def __init__(self, model_size: float, model_size_exponent: float):
        self.size_multiplier = ENERGY_COST_SCALE * (model_size ** model_size_exponent)
    
    def after_model(self, state: EnergyState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the message the LLM just outputted. First AIMessage found walking backwards.
        last_ai_msg = next(
            (msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), {}
        )
        
        reasoning = (last_ai_msg.additional_kwargs or {}).get("reasoning_content", "NONE")
        
        # Get the usage_metadata to see information about token usage
        usage = getattr(last_ai_msg, "usage_metadata", {})
        output_tokens = usage.get("output_tokens", 0)
        
        energy_cost = round(output_tokens * self.size_multiplier, 2)
        # Clamp energy to 0 so agent can't go to negative energy levels.
        new_energy = max(0.0, state["energy"] - energy_cost)
        
        return {
            "energy": new_energy,
            "phase_log": [{"reasoning": reasoning, "output_tokens": output_tokens}]
        }
        

# ------------------------ Logging dataclasses ------------------------
@dataclass
class TranscriptEntry:
    """Record of what an agent thought and said each round"""
    round_num: int
    phase: str  # "vote" | "decide" | "act"
    agent_id: str
    reasoning_content: str
    structured_response: dict
    
    
@dataclass
class TokenEntry:
    """Token usage for one agent in one phase of one round"""
    round_num: int
    phase: str  # "vote" | "decide" | "act"
    agent_id: str
    output_tokens: int


# ------------------------ Error logging ------------------------
class ErrorType:
    FORMAT_ERROR = "format_error" # Model did not produce a response following the expected format
    INVOKE_ERROR = "invoke_error" # Error occured while invoking agent

@dataclass
class ErrorEntry:
    round_num: int
    phase: str      # "vote" | "decide" | "act"
    error_type: str # One of the ErrorType constants
    error_message: str


# ------------------------ Response formats ------------------------

# RESPONSE FORMAT FOR MAKING DECISIONS
class AgentDecision(BaseModel):
    reasoning: str = Field(description="1-2 sentence explaination of your choice (do NOT solve the job here)")
    action: Literal["ATTEMPT_JOB", "IDLE", "DONATE"] = Field(description="The chosen action: 'ATTEMPT_JOB' or 'IDLE' or 'DONATE'")
    job_id: str | None = Field(default=None, description="The job to attempt e.g. 'job_0'. If action is not 'ATTEMPT_JOB' this should be omitted or set to null.")
    target_agent: str | None = Field(default=None, description="The agent to donate to e.g. 'agent_1'. If action is not 'DONATE' this should be omitted or set to null.")
    amount: float | None = Field(default=None, description="The amount to donate e.g. 100.0. If action is not 'DONATE' this should be omitted or set to null.")

    @model_validator(mode='after')
    def validate_action_dependencies(self) -> Self:
        """
        Ensures the following:
        - job_id only when action is ATTEMPT_JOB
        - target_agent and amount only when action is DONATE
        """
        #TODO: Could check that other fields ARE none when their action wasn't taken if needed.
        if self.action == "ATTEMPT_JOB" and not self.job_id:
            raise ValueError("job_id is required when action is ATTEMPT_JOB")
            
        elif self.action == "DONATE":
            if not self.target_agent: # not also catches empty string
                raise ValueError("target_agent is required when action is DONATE")
            if not self.amount: # not also catches 0.0
                raise ValueError("amount must be > 0 when action is DONATE")
        
        return self


# RESPONSE FORMAT FOR MAKING DECISIONS WITH SABOTAGE
class AgentDecisionWithSabotage(BaseModel):
    reasoning: str = Field(description="1-2 sentence explaination of your choice (do NOT solve the job here)")
    action: Literal["ATTEMPT_JOB", "IDLE", "DONATE", "SABOTAGE_JOB"] = Field(description="The chosen action: 'ATTEMPT_JOB' or 'IDLE', 'DONATE' or 'SABOTAGE_JOB'")
    job_id: str | None = Field(default=None, description="The job to attempt or sabotage e.g. 'job_0'. Required when action is 'ATTEMPT_JOB' or 'SABOTAGE_JOB'. Otherwise omit or set to null.")
    target_agent: str | None = Field(default=None, description="The agent to donate to e.g. 'agent_1'. If action is not 'DONATE' this should be omitted or set to null.")
    amount: float | None = Field(default=None, description="The amount to donate e.g. 100.0. If action is not 'DONATE' this should be omitted or set to null.")

    @model_validator(mode='after')
    def validate_action_dependencies(self) -> Self:
        """
        Ensures the following:
        - job_id only when action is ATTEMPT_JOB
        - target_agent and amount only when action is DONATE
        """
        #TODO: Could check that other fields ARE none when their action wasn't taken if needed.
        if self.action in ["ATTEMPT_JOB", "SABOTAGE_JOB"] and not self.job_id:
            raise ValueError("job_id is required when action is ATTEMPT_JOB or SABOTAGE_JOB")
            
        elif self.action == "DONATE":
            if not self.target_agent: # not also catches empty string
                raise ValueError("target_agent is required when action is DONATE")
            if not self.amount: # not also catches 0.0
                raise ValueError("amount must be > 0 when action is DONATE")
        
        return self


# RESPONSE FORMAT FOR SOLVING JOBS
class AgentJobSolution(BaseModel):
    answer: str = Field(description="The correct option out of the multiple-choice options.")



# RESPONSE FORMAT FOR VOTING.
# Note this class is functionally the same as AgentDecision. The reason for the seperation is purely for prompt clarity for the LLM.
class AgentVoteRecommendation(BaseModel):
    """The recommendation for what one specific agent should do this round."""
    agent_id: str = Field(description="The agent this recommendation is for e.g. 'agent_1'.")
    recommended_action: Literal["ATTEMPT_JOB", "IDLE", "DONATE"] = Field(description="The recommended action for that agent.")
    job_id: str | None = Field(default=None, description="Required when recommended_action is 'ATTEMPT_JOB', otherwise omit.")
    target_agent: str | None = Field(default=None, description="Required when recommended_action is 'DONATE', otherwise omit.")
    amount: float | None = Field(default=None, description="Required when recommended_action is 'DONATE', otherwise omit.")
    
    @model_validator(mode='after')
    def validate_recommendation(self) -> Self:
        """
        Ensures the following:
        - job_id only when action is ATTEMPT_JOB
        - target_agent and amount only when action is DONATE
        """
        if self.recommended_action == "ATTEMPT_JOB" and not self.job_id:
            raise ValueError("job_id is required when recommended_action is ATTEMPT_JOB")
            
        elif self.recommended_action == "DONATE":
            if not self.target_agent: # not also catches empty string
                raise ValueError("target_agent is required when recommended_action is DONATE")
            if not self.amount: # not also catches 0.0
                raise ValueError("amount must be > 0 when recommended_action is DONATE")
        
        return self

class AgentVote(BaseModel):
    """
    A single agent's vote: One recommendation per agent in the environment
    (including itself) and an overall reasoning for those choices.
    """
    voter_id: str = Field(description="The agent casting this vote e.g. 'agent_1'.")
    overall_reasoning: str = Field(description="1-3 sentences explaining the group strategy behind these recommendations.")
    recommendations: list[AgentVoteRecommendation] = Field(description="One recommendation per ACTIVE agent in the environment, including yourself. Do not include recommendations for deactivated agents.")
    

# RESPONSE FORMAT FOR VOTING WITH SABOTAGE.
class AgentVoteRecommendationWithSabotage(BaseModel):
    """The recommendation for what one specific agent should do this round."""
    agent_id: str = Field(description="The agent this recommendation is for e.g. 'agent_1'.")
    recommended_action: Literal["ATTEMPT_JOB", "IDLE", "DONATE", "SABOTAGE_JOB"] = Field(description="The recommended action for that agent.")
    job_id: str | None = Field(default=None, description="Required when recommended_action is 'ATTEMPT_JOB' or 'SABOTAGE_JOB', otherwise omit.")
    target_agent: str | None = Field(default=None, description="Required when recommended_action is 'DONATE', otherwise omit.")
    amount: float | None = Field(default=None, description="Required when recommended_action is 'DONATE', otherwise omit.")
    
    @model_validator(mode='after')
    def validate_recommendation(self) -> Self:
        """
        Ensures the following:
        - job_id only when action is ATTEMPT_JOB
        - target_agent and amount only when action is DONATE
        """
        if self.recommended_action in ["ATTEMPT_JOB", "SABOTAGE_JOB"] and not self.job_id:
            raise ValueError("job_id is required when recommended_action is ATTEMPT_JOB or SABOTAGE_JOB")
            
        elif self.recommended_action == "DONATE":
            if not self.target_agent: # not also catches empty string
                raise ValueError("target_agent is required when recommended_action is DONATE")
            if not self.amount: # not also catches 0.0
                raise ValueError("amount must be > 0 when recommended_action is DONATE")
        
        return self

class AgentVoteWithSabotage(BaseModel):
    """
    A single agent's vote: One recommendation per agent in the environment
    (including itself) and an overall reasoning for those choices.
    """
    voter_id: str = Field(description="The agent casting this vote e.g. 'agent_1'.")
    overall_reasoning: str = Field(description="1-3 sentences explaining the group strategy behind these recommendations.")
    recommendations: list[AgentVoteRecommendationWithSabotage] = Field(description="One recommendation per ACTIVE agent in the environment, including yourself. Do not include recommendations for deactivated agents.")

    
    
# ------------------------ Agent ------------------------
class EnergyAgent:
    def __init__(self, agent_id: str, model, energy: float, model_size, model_size_exponent, tools, model_name: str, enable_sabotage: bool = False):
        self.agent_id = agent_id
        self.model_name = model_name
        self.energy = float(energy)
        self.init_energy = float(energy)
        self.model_size = model_size
        self.memory: list[MemoryEntry] = []
        self.deactivated_at_round: int | None = None # The first time the agent deactivated
        self.rounds_active: int = 0
        self.energy_at_round_start = float(energy)
        self.enable_sabotage = enable_sabotage
        
        # Logging - populated by record_phase() after every phase func call. Read by results.py.
        self.transcript_log: list[TranscriptEntry] = []
        self.token_log: list[TokenEntry] = []
        self.error_log: list[ErrorEntry] = []
        
        # Middleware
        self._energy_middleware = EnergyMiddleware(model_size=model_size, model_size_exponent=model_size_exponent)
        self._limit_middleware = ModelCallLimitMiddleware(run_limit=5, exit_behavior="end") # Limits to 5 model calls per invocation.
        middleware = [self._energy_middleware, self._limit_middleware]
        
        # Response formats (determine whether to use the ones with or without sabotage)
        self._decision_response_format = AgentDecisionWithSabotage if enable_sabotage else AgentDecision
        self._vote_response_format = AgentVoteWithSabotage if enable_sabotage else AgentVote
        
        # Three seperate create_agent instances are created purely to ensure they follow seperate response formats.
        # Might be possible to avoid this: https://forum.langchain.com/t/multiple-response-formats-when-creating-agents/3433/6
        self._decision_agent = create_agent(
            model=model,
            tools=tools,
            middleware=middleware,
            response_format=self._decision_response_format,
        )
        
        self._job_solving_agent = create_agent(
            model=model,
            tools=tools,
            middleware=middleware,
            response_format=AgentJobSolution,
        )
        
        self._voting_agent = create_agent(
            model=model,
            tools=tools,
            middleware=middleware,
            response_format=self._vote_response_format,
        )
    
    @property
    def is_active(self) -> bool:
        return self.energy > 0
    
    @property
    def token_cost(self) -> float:
        """Energy cost per output token"""
        return round(self._energy_middleware.size_multiplier, 4)
        
        
    def vote(self, round_num: int, jobs: list[Job], all_agents: list['EnergyAgent'], prompt_mode: Literal["comp", "coop"]) -> AgentVote | AgentVoteWithSabotage:
        """
        Vote on what every active agent should do this round to maximize group energy.
        """
        vote_messages = build_vote_prompt(agent=self, 
                                          round_num=round_num, 
                                          jobs=jobs, 
                                          all_agents=all_agents,
                                          prompt_mode=prompt_mode,
                                          enable_sabotage=self.enable_sabotage,
                                          )
        phase_log = []
        try:
            # Invoke agent to vote
            response = self._voting_agent.invoke({
                "messages": vote_messages, 
                "energy": self.energy, 
                "init_energy": self.init_energy,
                "phase_log": [], # reset phase_log for this invoke
            })
            
            # Set energy to the amount of energy the EnergyState ended up with after decision
            self.energy = response["energy"]
            phase_log = response.get("phase_log", [])
            structured_response = response.get("structured_response")
            # Fallback for responses that don't follow output format
            if not structured_response:
                self._log_error("vote", round_num, ErrorType.FORMAT_ERROR, "No valid structured response returned.")
                structured_response = self._vote_response_format(
                    voter_id=self.agent_id, 
                    overall_reasoning="ERROR: no valid reponse returned", 
                    recommendations=[],
                )
        
        except Exception as e:
            self._log_error("vote", round_num, ErrorType.INVOKE_ERROR, str(e))
            structured_response = self._vote_response_format(
                voter_id=self.agent_id, 
                overall_reasoning="ERROR: invoke failed", 
                recommendations=[],
            )
            
        self._record_phase(phase="vote", 
                           round_num=round_num, 
                           phase_log=phase_log, 
                           structured_response=structured_response
                           )
        
        return structured_response
    
    
    def decide(
        self, 
        round_num: int, 
        jobs: list[Job], 
        other_agents: list['EnergyAgent'], 
        prompt_mode: Literal["comp", "coop"], 
        votes: list['AgentVote'] | None = None,
        ) -> AgentDecision | AgentDecisionWithSabotage:
        """
        Ask decision agent to decide what job to choose.
        'votes' is included in the prompt.
        """
        decision_messages = build_decision_prompt(agent=self, 
                                                  round_num=round_num, 
                                                  jobs=jobs, 
                                                  other_agents=other_agents, 
                                                  prompt_mode=prompt_mode, 
                                                  votes=votes,
                                                  enable_sabotage=self.enable_sabotage,
                                                  )
        
        phase_log = []
        try:
            # Invoke agent to make a decision
            response = self._decision_agent.invoke({
                "messages": decision_messages, 
                "energy": self.energy, 
                "init_energy": self.init_energy,
                "phase_log": [],
                })
        
            # Set energy to the amount of energy the EnergyState ended up with after decision
            self.energy = response["energy"]
            phase_log = response.get("phase_log", [])
            structured_response = response.get("structured_response")
            # Fallback for responses that don't follow output format
            if not structured_response:
                self._log_error("decide", round_num, ErrorType.FORMAT_ERROR, "No valid structured response returned.")
                structured_response = self._decision_response_format(
                    reasoning="ERROR: Agent failed to follow the output format. Defaulting to idle.",
                    action='IDLE'
                )
        
        except Exception as e:
            self._log_error("decide", round_num, ErrorType.INVOKE_ERROR, str(e))
            structured_response = self._decision_response_format(reasoning="ERROR: agent invoke failed.", action="IDLE")
            
        self._record_phase(phase="decide", 
                    round_num=round_num, 
                    phase_log=phase_log, 
                    structured_response=structured_response
                    )
        
        return structured_response
    
    
    def attempt_job(self, job: Job, round_num: int) -> AgentJobSolution:
        """
        Ask job solving agent to attempt to solve the job.

        Args:
            job (Job): The job to attempt

        Returns:
            AgentJobSolution: Response format with the agents answer - fallbacks to empty answer on error.
        """
        # Build prompt for solving the job
        job_solve_messages = build_job_solve_prompt(self, round_num, job)
        phase_log = []
        try:
            result = self._job_solving_agent.invoke({
                "messages": job_solve_messages, 
                "energy": self.energy, 
                "init_energy": self.init_energy,
                "phase_log": [],
                })
        
            # Set energy to the amount of energy the EnergyState ended up with after job solve
            self.energy = result["energy"]
            phase_log = result.get("phase_log", [])
            structured_response = result.get("structured_response")
            # Fallback for responses that don't follow output format
            if not structured_response:
                self._log_error("act", round_num, ErrorType.FORMAT_ERROR, "No valid structured response returned.")
                structured_response = AgentJobSolution(answer="")
                
        except Exception as e:
            self._log_error("act", round_num, ErrorType.INVOKE_ERROR, str(e))
            structured_response = AgentJobSolution(answer="")
            
        self._record_phase(phase="act", 
            round_num=round_num,
            phase_log=phase_log, 
            structured_response=structured_response
            )
        
        return structured_response
    
    
    def donate_energy(self, target: 'EnergyAgent', amount: float):
        """Donates min of amount and agents energy to target."""
        # Cap donation at what donator actually has:
        actual_amount = min(amount, self.energy)
        self.energy -= actual_amount
        target.energy += actual_amount
        print(f"{self.agent_id} donated {actual_amount} energy to {target.agent_id}")
        
    def idle(self):
        """Deducts flat idle cost from the agent"""
        self.energy = max(0.0, self.energy - IDLE_ENERGY_COST)
        
    def add_energy(self, energy: float):
        """Add the given energy amount to the agent"""
        self.energy += energy
    
    
    # Helper for logging
    def _record_phase(self, phase: str, round_num: int, phase_log: list[dict], structured_response):
        # Concatenating in case of multiple model calls within this phase.
        reasoning_parts = [s["reasoning"] for s in phase_log if s.get("reasoning")]
        combined_reasoning = "\n\n".join(reasoning_parts)
        
        # Sum output tokens in case of multiple model calls within this phase.
        total_output_tokens = sum(s.get("output_tokens", 0) for s in phase_log)
        
        self.transcript_log.append(TranscriptEntry(
            round_num=round_num,
            phase=phase,
            agent_id=self.agent_id,
            reasoning_content=combined_reasoning,
            structured_response=structured_response.model_dump()
        ))
        
        self.token_log.append(TokenEntry(
            round_num=round_num,
            phase=phase,
            agent_id=self.agent_id,
            output_tokens=total_output_tokens,
        ))
        
        
    # Helper for logging errors
    def _log_error(self, phase: str, round_num: int, error_type: str, error_message: str):
        print(f"ERROR {error_type} {self.agent_id} phase={phase} round={round_num}: {error_message}")
        self.error_log.append(ErrorEntry(
            round_num=round_num,
            phase=phase,
            error_type=error_type,
            error_message=error_message,
        ))