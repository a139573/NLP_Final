# Political Rhetoric NLP Pipeline

This repository contains the end-to-end Machine Learning pipeline for extracting, cleaning, translating, and analyzing political rhetoric using the ParlaMint-ES dataset. The project culminates in an NLP Master's Thesis, employing zero-shot LLM knowledge distillation into a high-speed RoBERTa classifier.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements & Installation](#requirements--installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)

## Project Overview
The pipeline processes parliamentary debates (ParlaMint), applies LLMs for attribute extraction, and distills this knowledge into faster, scalable models (RoBERTa). 

## Requirements & Installation
The project uses `uv` for dependency management, as defined in `pyproject.toml`. 

Ensure you have Python 3.12+ installed.
```bash
# If using uv
uv sync

# Or if using pip directly
pip install -r pyproject.toml
```

## Usage
The repository comes with a master orchestrator script `run_pipeline.py` which executes the full NLP pipeline from data extraction through final inference.

```bash
# Run the pipeline for a specific phase (1, 2, or 3)
python run_pipeline.py --phase 3

# Run a quick smoke-test (takes under 5 minutes) using toy mode
python run_pipeline.py --phase 3 --toy
```

## Pipeline Architecture
The master pipeline orchestrator executes 6 sequential steps:
1. **Data Extraction**: Extracts debate data from the ParlaMint corpus (`src/data/extract_parlamint.py`)
2. **Data Cleaning**: Cleans text and standardizes outputs (`src/data/clean_parlamint.py`)
3. **Feature Engineering**: Tokenizes and formats features for the RoBERTa model (`src/features/build_features_roberta.py`)
4. **Training / Hyperparameter Tuning**: Uses Optuna to tune RoBERTa parameters (`src/models/train_engine.py`)
5. **Evaluation & Production Retraining**: Evaluates performance and retrains the final model (`src/models/evaluate_model.py`)
6. **Inference & Reporting**: Runs inference and formats visualizations/reports (`src/visualization/inference_engine.py`)

## Project Structure
This structure adheres to the Cookiecutter Data Science standard:
```text
.
├── data/               # Raw, processed, and interim data
├── docs/               # Technical documentation
├── models/             # Trained and serialized models
├── notebooks/          # Jupyter notebooks for EDA and exploration
├── references/         # Data dictionaries and external references
├── reports/            # LaTeX thesis reports and generated figures
├── src/                # Pipeline source code
│   ├── data/           # Extraction and cleaning scripts
│   ├── features/       # Feature engineering and tokenization
│   ├── models/         # Training, evaluation, and optimization
│   └── visualization/  # Inference and reporting
├── run_pipeline.py     # Master orchestrator script
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # This file
```
