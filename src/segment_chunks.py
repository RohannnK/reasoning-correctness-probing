"""Segment chain-of-thought reasoning into chunks at transition keywords."""

import argparse
import re

from transformers import AutoTokenizer

from src.utils import load_jsonl, save_jsonl

TRANSITION_KEYWORDS = [
    "wait",
    "alternatively",
    "let me reconsider",
    "double-check",
    "hmm",
    "actually",
    "so the answer is",
]


def build_split_pattern(keywords: list[str]) -> re.Pattern:
    """Build a regex pattern that splits at transition keywords (case-insensitive)."""
    escaped = [re.escape(kw) for kw in keywords]
    pattern = r"(?i)(?=\b(?:" + "|".join(escaped) + r")\b)"
    return re.compile(pattern)


def extract_last_number(text: str) -> str | None:
    """Extract the last number appearing in a text chunk.

    Handles integers, decimals, and negative numbers. Strips commas from
    numbers like 1,234.
    """
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    return last


def normalize_number(s: str) -> str | None:
    """Normalize a number string for comparison."""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def get_token_offsets(tokenizer: AutoTokenizer, full_text: str) -> list[tuple[int, int]]:
    """Get character-to-token offset mapping.

    Returns a list of (start_char, end_char) for each token.
    """
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    return encoding["offset_mapping"]


def segment_cot(
    input_path: str,
    output_path: str,
    model_name: str,
    keywords: list[str] | None = None,
) -> None:
    """Segment CoT reasoning into chunks and assign correctness labels.

    Args:
        input_path: Path to the CoT JSONL file.
        output_path: Path to save the chunked JSONL file.
        model_name: HuggingFace model name (for tokenizer).
        keywords: Transition keywords to split on.
    """
    if keywords is None:
        keywords = TRANSITION_KEYWORDS

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    records = load_jsonl(input_path)
    split_pattern = build_split_pattern(keywords)

    all_chunks = []

    for record in records:
        problem_id = record["problem_id"]
        cot_text = record["generated_cot"]
        ground_truth = record["ground_truth_answer"]
        gt_normalized = normalize_number(ground_truth)

        # Split at transition keywords
        parts = split_pattern.split(cot_text)
        parts = [p for p in parts if p.strip()]

        if not parts:
            parts = [cot_text]

        # Get token offsets for the full CoT
        offset_mapping = get_token_offsets(tokenizer, cot_text)

        char_pos = 0
        for chunk_id, chunk_text in enumerate(parts):
            chunk_start_char = cot_text.find(chunk_text, char_pos)
            chunk_end_char = chunk_start_char + len(chunk_text)
            char_pos = chunk_end_char

            # Find token range for this chunk
            token_start = None
            token_end = None
            for tok_idx, (ts, te) in enumerate(offset_mapping):
                if ts >= chunk_start_char and token_start is None:
                    token_start = tok_idx
                if te <= chunk_end_char:
                    token_end = tok_idx

            if token_start is None:
                token_start = 0
            if token_end is None:
                token_end = len(offset_mapping) - 1

            intermediate_answer = extract_last_number(chunk_text)
            ia_normalized = normalize_number(intermediate_answer)
            is_correct = (ia_normalized == gt_normalized) if ia_normalized is not None else False

            all_chunks.append({
                "problem_id": problem_id,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "chunk_token_start": token_start,
                "chunk_token_end": token_end,
                "intermediate_answer": intermediate_answer,
                "is_correct": is_correct,
                "ground_truth": ground_truth,
            })

    save_jsonl(all_chunks, output_path)
    print(f"Saved {len(all_chunks)} chunks from {len(records)} problems to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment CoT reasoning into chunks")
    parser.add_argument("--input_path", type=str, required=True, help="Input CoT JSONL path")
    parser.add_argument("--output_path", type=str, required=True, help="Output chunks JSONL path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (for tokenizer)")
    args = parser.parse_args()

    segment_cot(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
