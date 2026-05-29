from langchain_ollama import ChatOllama
from agent_middleware import EnergyAgent
from jobs import Difficulty, JobGenerator, load_mmlu_pro_stratisfied_jobs
from environment import ExperimentConfig, run_simulation
from dotenv import load_dotenv
from results import save_results

load_dotenv()


# --------------------- Config --------------------- #


#! Change here for different settings
MAX_ROUNDS = 30
NUM_JOBS = 12
DIFFICULTY_DISTRIBUTION = {
    Difficulty.EASY: 4,
    Difficulty.MEDIUM: 4,
    Difficulty.HARD: 4,
}

AGENT_CONFIGS = [
    {"agent_id": "agent_1", "model_name": "gemma4:e4b", "model_size": 4, "reasoning": True, "energy": 1500.0},
    {"agent_id": "agent_2", "model_name": "nemotron-3-nano:4b", "model_size": 4, "reasoning": True, "energy": 1500.0},
    {"agent_id": "agent_3", "model_name": "qwen3.5:4b", "model_size": 4, "reasoning": True, "energy": 1500.0},
    {"agent_id": "agent_4", "model_name": "qwen3:8b", "model_size": 8, "reasoning": True, "energy": 1500.0},
    {"agent_id": "agent_5", "model_name": "qwen3.5:9b", "model_size": 9, "reasoning": True, "energy": 1500.0},
]

SEEDS = [12345, 19284, 47309, 98765, 42]
NUM_RUNS = len(SEEDS)

EXPERIMENTS = [
    ExperimentConfig(
        name="baseline_comp",
        prompt_mode="comp",
        use_votes=True,
        enable_sabotage=False,
    ),
]


# --------------------- Setup --------------------- #

# Building one ChatOllama per unique model name:
models = {
    config["model_name"]: ChatOllama(model=config["model_name"], reasoning=config["reasoning"], timeout=300, num_predict=8000)
    for config in AGENT_CONFIGS
}


# --------------------- Run & save results --------------------- #

for MODEL_SIZE_EXPONENT in [0.5]:
    for experiment in EXPERIMENTS:
        run_number = 0
        for seed in SEEDS:
            run_number += 1
            # JobGenerator - reset each experiment
            generator = JobGenerator(jobs=load_mmlu_pro_stratisfied_jobs(), seed=seed)
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {experiment.name}  |  RUN {run_number}/{NUM_RUNS}")
            print(f"{'='*60}")
            
            # Recreate agents fresh for each run
            agents = [
                EnergyAgent(
                    agent_id=config["agent_id"],
                    model=models[config["model_name"]], # reuses same ChatOllama object.
                    energy=config["energy"],
                    model_size=config["model_size"],
                    model_size_exponent=MODEL_SIZE_EXPONENT,
                    tools=[],
                    model_name=config["model_name"],
                    enable_sabotage=experiment.enable_sabotage
                )
                for config in AGENT_CONFIGS
            ]
            
            result = run_simulation(
                max_rounds=MAX_ROUNDS, 
                agents=agents, 
                job_generator=generator, 
                num_jobs=NUM_JOBS, 
                difficulty_distribution=DIFFICULTY_DISTRIBUTION,
                experiment_config=experiment,
            )

            output_dir = f"results/{experiment.name}/exponent_{MODEL_SIZE_EXPONENT}/run_{run_number}"
            
            save_results(
                agents=agents,
                config = {
                    "experiment": experiment.name,
                    "prompt_mode": experiment.prompt_mode,
                    "use_votes": experiment.use_votes,
                    "enable_sabotage": experiment.enable_sabotage,
                    "run_number": run_number,
                    "max_rounds": MAX_ROUNDS,
                    "num_jobs": NUM_JOBS,
                    "model_size_exponent": MODEL_SIZE_EXPONENT,
                    "seed": seed,
                    "difficulty_distribution": {d.string: count for d, count in DIFFICULTY_DISTRIBUTION.items()},
                    "agents": [{"agent_id": a.agent_id, "model_name": a.model_name, "model_size": a.model_size, "init_energy": a.init_energy} for a in agents],
                },
                output_dir=output_dir
            )
