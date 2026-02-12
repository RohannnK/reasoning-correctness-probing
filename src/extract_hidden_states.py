"""Extract hidden states from model forward passes at chunk boundaries."""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_jsonl

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_MODEL_KEY = "distilled"
DEFAULT_CHUNKS_PATH = "data/chunks/gsm8k_chunks.jsonl"


def char_to_token_pos(offsets: list[tuple[int, int]], char_end: int) -> int:
    """Map a character position to the corresponding token index.

    Args:
        offsets: List of (char_start, char_end) tuples from the tokenizer's
            offset_mapping, one per token.
        char_end: The character position to map.

    Returns:
        Token index whose span contains or is closest to char_end.
    """
    best_idx = len(offsets) - 1  # fallback: last token

    for i, (tok_start, tok_end) in enumerate(offsets):
        # Skip special-token entries that map to (0, 0)
        if tok_start == 0 and tok_end == 0 and i > 0:
            continue

        # Exact containment: char_end falls within this token's span
        if tok_start < char_end <= tok_end:
            return i

        # char_end is between this token and the next (gap between tokens)
        if tok_end <= char_end:
            best_idx = i

    return best_idx


def extract_hidden_states(
    model_name: str,
    model_key: str,
    chunks_path: str,
) -> None:
    """Extract hidden state vectors at the last token of each chunk boundary.

    Processes one problem at a time to manage GPU memory. For each problem,
    tokenizes the full CoT with offset mapping, runs a forward pass with
    output_hidden_states=True, and extracts the hidden state at each chunk's
    boundary token from every layer.

    Args:
        model_name: HuggingFace model name or path.
        model_key: Key for output subdirectory (e.g. 'base' or 'distilled').
        chunks_path: Path to the chunked JSONL file.
    """
    output_dir = os.path.join("data", "hidden_states", model_key)
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"Loading chunks from: {chunks_path}")
    problems = load_jsonl(chunks_path)
    total = len(problems)
    print(f"Found {total} problems")

    processed = 0
    skipped = 0
    total_hidden_states = 0

    for idx, problem in enumerate(problems):
        problem_id = problem["problem_id"]
        save_path = os.path.join(output_dir, f"problem_{problem_id}.npz")

        # Resume support: skip already-saved problems
        if os.path.exists(save_path):
            skipped += 1
            continue

        full_output = problem["full_output"]
        chunks = problem["chunks"]

        # Tokenize with offset mapping for char-to-token alignment
        encoding = tokenizer(
            full_output,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].to(model.device)
        offsets = encoding["offset_mapping"][0].tolist()  # list of (start, end)
        seq_len = input_ids.shape[1]

        # Map each chunk's char_end to a token position
        token_positions = []
        labels = []
        for chunk in chunks:
            tok_pos = char_to_token_pos(offsets, chunk["char_end"])
            tok_pos = min(tok_pos, seq_len - 1)
            token_positions.append(tok_pos)
            labels.append(chunk["is_correct"])

        labels_arr = np.array(labels, dtype=bool)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
        num_layers = len(hidden_states)

        # Extract hidden state at each chunk boundary from all layers
        arrays = {}
        for layer_idx in range(num_layers):
            layer_hidden = hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
            vecs = []
            for tok_pos in token_positions:
                vecs.append(layer_hidden[0, tok_pos, :].cpu().float().numpy())
            arrays[f"layer_{layer_idx}"] = np.stack(vecs, axis=0)

        arrays["labels"] = labels_arr
        np.savez_compressed(save_path, **arrays)

        processed += 1
        total_hidden_states += len(chunks) * num_layers

        # Free GPU memory
        del outputs, hidden_states, input_ids, encoding

        # Clear CUDA cache every 50 problems
        if processed % 50 == 0:
            torch.cuda.empty_cache()

        # Log progress every 10 problems
        if processed % 10 == 0:
            print(f"[{processed}/{total - skipped}] Processed problem {problem_id}")

    if skipped > 0:
        print(f"Skipped {skipped} already-saved problems")
    print(f"Total problems processed: {processed}")
    print(f"Total hidden states saved: {total_hidden_states}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract hidden states at chunk boundaries")
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model_key", type=str, default=DEFAULT_MODEL_KEY,
        help=f"Output subdirectory key (default: {DEFAULT_MODEL_KEY})",
    )
    parser.add_argument(
        "--chunks_path", type=str, default=DEFAULT_CHUNKS_PATH,
        help=f"Path to chunked JSONL (default: {DEFAULT_CHUNKS_PATH})",
    )
    args = parser.parse_args()

    extract_hidden_states(
        model_name=args.model_name,
        model_key=args.model_key,
        chunks_path=args.chunks_path,
    )


if __name__ == "__main__":
    main()
