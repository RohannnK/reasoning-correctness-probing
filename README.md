# Reasoning Correctness Probing

Probing hidden states of language models to detect correctness signals in chain-of-thought reasoning. Compares a base model (Qwen2.5-Math-1.5B) against its reasoning-distilled variant (DeepSeek-R1-Distill-Qwen-1.5B) to determine whether reasoning distillation creates new correctness-encoding representations or surfaces pre-existing ones.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Generate Chain-of-Thought

```bash
# Base model
python -m src.generate_cot \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --output_path data/cot/qwen-base.jsonl

# Distilled model
python -m src.generate_cot \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_path data/cot/r1-distill.jsonl
```

### 2. Segment into Chunks

```bash
python -m src.segment_chunks \
    --input_path data/cot/qwen-base.jsonl \
    --output_path data/chunks/qwen-base.jsonl \
    --model_name Qwen/Qwen2.5-Math-1.5B

python -m src.segment_chunks \
    --input_path data/cot/r1-distill.jsonl \
    --output_path data/chunks/r1-distill.jsonl \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

### 3. Extract Hidden States

```bash
python -m src.extract_hidden_states \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --chunks_path data/chunks/qwen-base.jsonl \
    --output_dir data/hidden_states \
    --model_short_name qwen-base

python -m src.extract_hidden_states \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --chunks_path data/chunks/r1-distill.jsonl \
    --output_dir data/hidden_states \
    --model_short_name r1-distill
```

### 4. Train Probes

```bash
python -m src.train_probes \
    --hidden_states_dir data/hidden_states \
    --model_short_name qwen-base \
    --output_dir results/probes

python -m src.train_probes \
    --hidden_states_dir data/hidden_states \
    --model_short_name r1-distill \
    --output_dir results/probes
```

### 5. Analysis

```bash
# CKA
python -m src.cka_analysis \
    --hidden_states_dir data/hidden_states \
    --base_short_name qwen-base \
    --distilled_short_name r1-distill \
    --output_path results/cka_results.csv

# Sparsity
python -m src.sparsity_analysis \
    --hidden_states_dir data/hidden_states \
    --model_short_name qwen-base \
    --output_path results/qwen-base_sparsity.csv

python -m src.sparsity_analysis \
    --hidden_states_dir data/hidden_states \
    --model_short_name r1-distill \
    --output_path results/r1-distill_sparsity.csv

# Early exit
python -m src.early_exit \
    --hidden_states_dir data/hidden_states \
    --probes_dir results/probes \
    --model_short_name qwen-base \
    --output_path results/qwen-base_early_exit.csv

python -m src.early_exit \
    --hidden_states_dir data/hidden_states \
    --probes_dir results/probes \
    --model_short_name r1-distill \
    --output_path results/r1-distill_early_exit.csv
```

### 6. Visualize

```bash
python -m src.visualize \
    --results_dir results \
    --figures_dir results/figures
```

## Project Structure

```
├── configs/experiment_config.yaml   # Model names, hyperparameters
├── src/
│   ├── generate_cot.py              # CoT generation
│   ├── segment_chunks.py            # Chunk segmentation + labeling
│   ├── extract_hidden_states.py     # Hidden state extraction
│   ├── train_probes.py              # Linear & MLP probe training
│   ├── cka_analysis.py              # Cross-model CKA comparison
│   ├── sparsity_analysis.py         # L1 sparsity analysis
│   ├── early_exit.py                # Early exit simulation
│   ├── visualize.py                 # Figure generation
│   └── utils.py                     # Shared utilities
├── data/                            # Generated data (gitignored)
└── results/                         # Outputs, probes, figures
```
