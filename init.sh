#!/bin/bash
set -e

# System packages
sudo apt-get update
sudo apt-get install -y zstd

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > ~/ollama.log 2>&1 &

# Wait for Ollama server to be ready
echo "Waiting for Ollama server..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama server is ready."

# Pull models
ollama pull qwen3:8b; ollama pull qwen3.5:4b; ollama pull gemma4:e4b; ollama pull nemotron-3-nano:4b; ollama pull qwen3.5:9b

# Install Python dependencies
pip install -r requirements.txt

# Run experiment
python main.py