"""Generate chain-of-thought reasoning for GSM8K problems using a HuggingFace model."""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OUTPUT = "data/cot_outputs/gsm8k_cot.jsonl"


def extract_ground_truth(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K answer field (after ####)."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip()


def extract_model_answer(full_output: str) -> str:
    """Extract the model's final answer from text after </think>."""
    parts = full_output.split("</think>")
    if len(parts) < 2:
        answer_region = full_output
    else:
        answer_region = parts[-1]
    # Find the last number (possibly negative, with commas/decimals)
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_region)
    if numbers:
        return numbers[-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison: strip $, %, commas, whitespace."""
    answer = answer.strip()
    answer = answer.replace(",", "").replace("$", "").replace("%", "")
    answer = answer.strip()
    return answer


def build_prompt(question: str) -> str:
    """Build ChatML prompt with <think> trigger for R1-Distill models."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def count_existing_lines(path: str) -> int:
    """Count lines in an existing file for resume support."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def generate_cot(
    model_name: str,
    output_path: str,
    num_problems: int,
    batch_size: int,
    split: str,
    temperature: float,
    max_new_tokens: int,
) -> None:
    """Generate chain-of-thought reasoning and save as JSONL."""
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
    tokenizer.padding_side = "left"

    print(f"Loading dataset: gsm8k/main [{split}]")
    dataset = load_dataset("gsm8k", "main", split=split)
    num_problems = min(num_problems, len(dataset))
    dataset = dataset.select(range(num_problems))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Resume support
    skip = count_existing_lines(output_path)
    if skip > 0:
        print(f"Resuming: skipping {skip} already-completed problems")
    if skip >= num_problems:
        print("All problems already completed.")
        return

    file_mode = "a" if skip > 0 else "w"
    correct_count = 0
    total_count = 0

    # Count correct answers from already-written lines for accurate running stats
    if skip > 0:
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("is_correct"):
                    correct_count += 1
                total_count += 1

    remaining_indices = list(range(skip, num_problems))

    with open(output_path, file_mode) as f_out:
        for batch_start in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[batch_start : batch_start + batch_size]
            batch_examples = [dataset[i] for i in batch_indices]

            prompts = [build_prompt(ex["question"]) for ex in batch_examples]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                )

            for j, idx in enumerate(batch_indices):
                input_len = inputs["input_ids"].shape[1]
                generated_ids = outputs[j][input_len:]
                full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

                ground_truth = extract_ground_truth(batch_examples[j]["answer"])
                model_answer = extract_model_answer(full_output)
                is_correct = normalize_answer(model_answer) == normalize_answer(ground_truth)

                total_count += 1
                if is_correct:
                    correct_count += 1

                record = {
                    "problem_id": idx,
                    "question": batch_examples[j]["question"],
                    "ground_truth": ground_truth,
                    "full_output": full_output,
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

                if total_count % 10 == 0:
                    acc = correct_count / total_count * 100
                    print(f"[{total_count}/{num_problems}] Running accuracy: {acc:.1f}%")

    acc = correct_count / total_count * 100 if total_count > 0 else 0.0
    print(f"\nDone. {total_count} problems, {correct_count} correct, accuracy: {acc:.1f}%")
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chain-of-thought reasoning for GSM8K")
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output_path", type=str, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--num_problems", type=int, default=500, help="Number of problems to generate (default: 500)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation (default: 4)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    generate_cot(
        model_name=args.model_name,
        output_path=args.output_path,
        num_problems=args.num_problems,
        batch_size=args.batch_size,
        split=args.split,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
