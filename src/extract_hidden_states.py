"""Extract hidden states from model forward passes at chunk boundaries."""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_jsonl


def extract_hidden_states(
    model_name: str,
    chunks_path: str,
    output_dir: str,
    model_short_name: str,
) -> None:
    """Extract hidden state vectors at the last token of each chunk.

    Processes one problem at a time to manage GPU memory. For each problem,
    tokenizes the full CoT, runs a forward pass with output_hidden_states=True,
    and extracts the hidden state at each chunk's last token from every layer.

    Args:
        model_name: HuggingFace model name or path.
        chunks_path: Path to the chunked JSONL file.
        output_dir: Directory to save compressed npz files.
        model_short_name: Short name for filename prefix.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = load_jsonl(chunks_path)

    # Group chunks by problem_id
    problems: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        problems[chunk["problem_id"]].append(chunk)

    os.makedirs(output_dir, exist_ok=True)

    for problem_id, problem_chunks in tqdm(problems.items(), desc="Extracting hidden states"):
        problem_chunks.sort(key=lambda c: c["chunk_id"])

        # Reconstruct the full CoT text from the first problem record
        # We need the original CoT to get proper tokenization
        full_cot = "".join(c["chunk_text"] for c in problem_chunks)
        inputs = tokenizer(full_cot, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)

        seq_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
        num_layers = len(hidden_states)

        # Extract hidden state at the last token of each chunk
        arrays: dict[str, np.ndarray] = {}
        labels = []

        for chunk in problem_chunks:
            token_end = min(chunk["chunk_token_end"], seq_len - 1)
            labels.append(int(chunk["is_correct"]))

        labels_arr = np.array(labels, dtype=np.int64)

        for layer_idx in range(num_layers):
            layer_hidden = hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
            layer_vectors = []

            for chunk in problem_chunks:
                token_end = min(chunk["chunk_token_end"], seq_len - 1)
                vec = layer_hidden[0, token_end, :].cpu().float().numpy()
                layer_vectors.append(vec)

            arrays[f"layer_{layer_idx}"] = np.stack(layer_vectors, axis=0)

        arrays["labels"] = labels_arr

        # Save as compressed npz
        save_path = os.path.join(output_dir, f"{model_short_name}_problem_{problem_id}.npz")
        np.savez_compressed(save_path, **arrays)

        # Free GPU memory
        del outputs, hidden_states, input_ids
        torch.cuda.empty_cache()

    print(f"Saved hidden states for {len(problems)} problems to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract hidden states at chunk boundaries")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--chunks_path", type=str, required=True, help="Chunked JSONL path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for npz files")
    parser.add_argument("--model_short_name", type=str, required=True, help="Short name for filenames")
    args = parser.parse_args()

    extract_hidden_states(
        model_name=args.model_name,
        chunks_path=args.chunks_path,
        output_dir=args.output_dir,
        model_short_name=args.model_short_name,
    )


if __name__ == "__main__":
    main()
