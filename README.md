# Reasoning Correctness Probing

Probing hidden states of language models to detect correctness signals during chain-of-thought (CoT) reasoning. We compare a base model (Qwen2.5-Math-1.5B) against its reasoning-distilled variant (DeepSeek-R1-Distill-Qwen-1.5B) using a same-input design: both models process the distilled model's CoT text, so hidden state differences reflect representational geometry rather than input differences. Linear and MLP probes trained on 2500 GSM8K problems show that both models encode correctness above chance at every layer, with nearly identical peak accuracy (base 0.858 vs distilled 0.856 at layer 17). A native baseline control (base model on its own CoT) yields AUC 0.789, confirming the distilled `<think>` format provides a probing boost — the OOD format is not suppressing the base model's signal but enhancing it.

## Key Results

| Metric | Value |
|--------|-------|
| Base model (same-input, L17) | ROC-AUC = 0.858 [0.81, 0.91] |
| Distilled model (same-input, L17) | ROC-AUC = 0.852 [0.81, 0.89] |
| Distilled best (L19) | ROC-AUC = 0.855 [0.81, 0.90] |
| Native baseline (base own CoT, L18) | ROC-AUC = 0.789 [0.72, 0.86] |
| Position-only baseline | ROC-AUC = 0.850 |
| Length-only baseline | ROC-AUC = 0.506 |

**Position confound finding:** A position-only logistic regression achieves AUC 0.850 using only relative chunk position (later chunks correlate with correctness). All probe results reported above use **last-chunk-only** evaluation, which controls for position by using only the final chunk of each problem. Under this control, probes still exceed chance, confirming a genuine correctness signal beyond positional artifacts.

**Native baseline (OOD control):** The base model probed on its own native CoT (AUC 0.789) scores ~0.07 below same-input probing (AUC 0.858). The distilled `<think>` format helps rather than hurts base model probing. This means the base–distilled AUC gap in the main experiment is partly a format effect, not purely representational geometry.

## Reproducing Results

```bash
pip install -r requirements.txt

# Full pipeline (requires GPU + Kaggle for hidden state extraction)
python -m src.generate_cot          # Generate CoT outputs
python -m src.segment_chunks        # Segment into labeled chunks
python -m src.extract_hidden_states # Extract hidden states (GPU required)

# Analysis (runs on CPU, uses pre-extracted hidden states)
python -m src.train_probes          # Train probes (last-chunk-only, 10-split CV)
python -m src.cka_analysis          # CKA cross-model comparison + permutation tests
python -m src.sparsity_analysis     # L1 probes, dimension overlap analysis
python -m src.early_exit            # Early exit simulation
python -m src.length_confound_check # Length-only baseline validation
python -m src.native_baseline_2500  # Native baseline OOD control (2500 problems)
python -m src.visualize             # Generate all figures
```

Hidden states are not included in the repository (~2 GB total). They can be regenerated using the extraction scripts on Kaggle with GPU access, or by running `src/extract_hidden_states.py` locally with a CUDA-capable GPU.

## Repository Structure

```
├── configs/experiment_config.yaml        # Model names, hyperparameters
├── data/
│   ├── cot_outputs/                      # Raw CoT generation outputs (JSONL)
│   ├── chunks/                           # Segmented + labeled chunks (JSONL)
│   └── hidden_states/                    # Extracted hidden states (gitignored, ~2GB)
├── src/
│   ├── generate_cot.py                   # CoT generation (distilled model)
│   ├── segment_chunks.py                 # Chunk segmentation + labeling
│   ├── extract_hidden_states.py          # Hidden state extraction
│   ├── train_probes.py                   # Linear & MLP probe training
│   ├── cka_analysis.py                   # CKA + permutation tests
│   ├── sparsity_analysis.py              # L1 probes, dimension overlap
│   ├── early_exit.py                     # Early exit simulation
│   ├── length_confound_check.py          # Length-only baseline validation
│   ├── analyze_native_baseline.py        # Native baseline (500 problems, legacy)
│   ├── native_baseline_2500.py           # Native baseline (2500 problems)
│   ├── visualize.py                      # Figure generation
│   └── utils.py                          # Shared utilities
├── results/
│   ├── metrics/                          # CSV/JSON result files
│   └── figures/                          # Generated figures (PNG)
├── requirements.txt
└── README.md
```

## Citation

**Probing for Correctness: How Reasoning Distillation Reshapes Hidden Representations**
Rohan Kaman
```
```
