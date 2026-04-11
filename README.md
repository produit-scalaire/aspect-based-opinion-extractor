Aspect-Based Opinion Extraction 

## 1. Project Deliverable Information
* **Authors:** Paul-Emile Barranger, Valentin Dugay, Clément Callaert
* **Course:** NLP Course @ CentraleSupélec (2025-2026)

## 2. Project Overview
This repository implements an Aspect-Based Sentiment Analysis (ABSA) system for French restaurant reviews. The objective is to extract the overall opinion across three predefined aspects: **Price**, **Food**, and **Service**. 

## 3. Ollama and running the project (local setup)

**Install Ollama (Linux):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Download the LLM** (use the same name as `ollama_model` in `src/config.py`; default below matches that field):
```bash
ollama pull gemma3:4b
```

**Start the Ollama server** (keep this terminal open while you run the project):
```bash
ollama serve
```
If Ollama was installed as a system service and is already running, you can skip this step.

**Stop the Ollama server:**
- If you started it with `ollama serve` in a terminal: press **Ctrl+C** in that terminal.
- If it runs as a background service: `sudo systemctl stop ollama` (only when applicable on your machine).

**Run the project** (from the repository root, with Ollama listening on the default URL):
```bash
cd src
accelerate launch runproject.py --ollama_url=http://localhost:11434/v1
```
If your Ollama API uses another host or port, change `--ollama_url` accordingly.
