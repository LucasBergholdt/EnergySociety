# Energy-Constrained Multi-Agent Simulation

This repository contains a simulation framework built via LangChain/LangGraph for comparing competitive and cooperative behaviour in an energy-constrained multi-agent system. Agents are powered by local Ollama models, spend energy when generating tokens, choose between attempting jobs, idling, or donating energy, and receive rewards when they solve multiple-choice jobs correctly.

Experiments use questions from the MMLU-Pro-Stratified dataset and support both a competitive (`comp`) and cooperative (`coop`) setting.

## Repository overview

| File | Purpose |
| --- | --- |
| `main.py` | Main entry point. Configure experiments, agents, seeds, models, difficulty distribution, and run experiments here. |
| `environment.py` | Defines the LangGraph simulation flow for competitive and cooperative experiments. |
| `agent_middleware.py` | Defines `EnergyAgent`, structured response schemas, token/energy accounting middleware, and logging of agent phases. |
| `prompts.py` | Contains the prompts used for decision-making, job solving, and cooperative voting. |
| `jobs.py` | Defines jobs, difficulties, job evaluation, and dataset loading from MMLU-Pro-Stratified. |
| `memory.py` | Defines per-round memory entries stored by each agent. |
| `results.py` | Saves run outputs to JSON files. |
| `init.sh` | Bash setup script for installing Ollama, pulling models, installing requirements, and running the experiment. |
| `plotting/` | Optional analysis/plotting utilities for completed experiments. |

## Requirements

- Ollama installed and running 
- Python dependencies are listed in `requirements.txt`.
- Internet access, or a cached copy of the MMLU-Pro-Stratified dataset, is needed when the dataset is loaded for the first time.


## Quick start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama from [here](https://ollama.com/download).


Start Ollama in a seperate terminal:
```bash
ollama serve
```

Pull the relevant Ollama models -- here default models used in `main.py` are shown:
```bash
ollama pull gemma4:e4b
ollama pull nemotron-3-nano:4b
ollama pull qwen3.5:4b
ollama pull qwen3:8b
ollama pull qwen3.5:9b
```

Run the experiment:

```bash
python main.py
```

Results are saved under:

```text
results/{experiment.name}/exponent_{model_size_exponent}/run_{run_number}
```

Each run writes `agents.json`, `config.json`, `transcript.json`, and `token_usage.json`.


## Ollama

Install Ollama from <https://ollama.com/download>

Start the Ollama server and keep it running while the experiment runs:

```bash
ollama serve
```

Pull the required Ollama models.
The default `main.py` configuration uses the following models:

```bash
ollama pull gemma4:e4b
ollama pull nemotron-3-nano:4b
ollama pull qwen3.5:4b
ollama pull qwen3:8b
ollama pull qwen3.5:9b
```

The model names in `main.py` must match models available in your local Ollama installation. 
If a model name is unavailable, either pull the correct model or edit `AGENT_CONFIGS` in `main.py` to use models you have installed.


## Optional `.env` file

`main.py` calls `load_dotenv()`, so you may add a `.env` file if you want LangSmith tracing.
The current local Ollama setup does not require one by default.