"""Segment chain-of-thought reasoning into chunks at paragraph boundaries and transition keywords."""

import argparse
import re

from src.utils import load_jsonl, save_jsonl

DEFAULT_INPUT = "data/cot_outputs/gsm8k_cot.jsonl"
DEFAULT_OUTPUT = "data/chunks/gsm8k_chunks.jsonl"

TRANSITION_KEYWORDS = [
    "wait",
    "alternatively",
    "let me reconsider",
    "let me verify",
    "double-check",
    "hmm",
    "actually",
    "but ",
    "however",
    "let me try",
    "so the answer is",
]


def build_split_pattern(keywords: list[str]) -> re.Pattern:
    """Build a regex pattern that splits at transition keywords (case-insensitive).

    Uses a lookahead so the keyword text stays in the chunk that follows the split.
    Keywords with trailing spaces (like "but ") use a space boundary instead of \\b.
    """
    parts = []
    for kw in keywords:
        escaped = re.escape(kw)
        if kw.endswith(" "):
            # "but " — require word boundary before, trailing space is literal
            parts.append(r"\b" + escaped)
        else:
            parts.append(r"\b" + escaped + r"\b")
    pattern = r"(?i)(?=" + "|".join(parts) + r")"
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


def segment_cot(
    input_path: str,
    output_path: str,
    keywords: list[str] | None = None,
) -> None:
    """Segment CoT reasoning into chunks and assign correctness labels.

    Args:
        input_path: Path to the CoT JSONL file.
        output_path: Path to save the chunked JSONL file.
        keywords: Transition keywords to split on.
    """
    if keywords is None:
        keywords = TRANSITION_KEYWORDS

    records = load_jsonl(input_path)
    keyword_pattern = build_split_pattern(keywords)
    # Primary split: double newlines (paragraph boundaries)
    primary_pattern = re.compile(r"\n\n")

    output_records = []
    total_chunks = 0
    correct_chunks = 0
    single_chunk_problems = 0

    for record in records:
        problem_id = record["problem_id"]
        question = record["question"]
        ground_truth = record["ground_truth"]
        full_output = record["full_output"]
        gt_normalized = normalize_number(ground_truth)

        # Strip content after </think> (formatted answer is redundant)
        think_end = full_output.find("</think>")
        reasoning = full_output[:think_end] if think_end != -1 else full_output

        # Primary split on paragraph boundaries (double newlines)
        primary_parts = primary_pattern.split(reasoning)
        # Secondary split: within each primary chunk, split at transition keywords
        parts = []
        for pp in primary_parts:
            sub = keyword_pattern.split(pp)
            parts.extend(sub)
        parts = [p for p in parts if p.strip()]

        if not parts:
            parts = [reasoning]

        chunks = []
        char_pos = 0
        for chunk_id, chunk_text in enumerate(parts):
            char_start = full_output.find(chunk_text, char_pos)
            char_end = char_start + len(chunk_text)
            char_pos = char_end

            intermediate_answer = extract_last_number(chunk_text)
            ia_normalized = normalize_number(intermediate_answer)
            is_correct = (ia_normalized == gt_normalized) if ia_normalized is not None else False

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
                "intermediate_answer": intermediate_answer,
                "is_correct": is_correct,
            })

            total_chunks += 1
            if is_correct:
                correct_chunks += 1

        if len(chunks) == 1:
            single_chunk_problems += 1

        output_records.append({
            "problem_id": problem_id,
            "question": question,
            "ground_truth": ground_truth,
            "full_output": full_output,
            "chunks": chunks,
        })

    save_jsonl(output_records, output_path)

    # Print summary stats
    num_problems = len(output_records)
    avg_chunks = total_chunks / num_problems if num_problems else 0
    pct_correct = 100 * correct_chunks / total_chunks if total_chunks else 0
    pct_incorrect = 100 - pct_correct if total_chunks else 0

    print(f"Saved {total_chunks} chunks from {num_problems} problems to {output_path}")
    print(f"  Total chunks:          {total_chunks}")
    print(f"  Avg chunks per problem: {avg_chunks:.2f}")
    print(f"  Class balance:         {pct_correct:.1f}% correct / {pct_incorrect:.1f}% incorrect")
    print(f"  Problems with 1 chunk: {single_chunk_problems}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment CoT reasoning into chunks")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT,
                        help="Input CoT JSONL path")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT,
                        help="Output chunks JSONL path")
    args = parser.parse_args()

    segment_cot(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
