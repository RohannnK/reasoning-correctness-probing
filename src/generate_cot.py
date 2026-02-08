"""Generate chain-of-thought reasoning for GSM8K problems using a HuggingFace model."""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_ground_truth(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K answer field.

    GSM8K answers end with '#### <number>'.
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip()


def build_prompt(problem: str, is_r1_distill: bool) -> str:
    """Build the generation prompt for a given problem.

    For R1-Distill models, prepend <think> to trigger reasoning mode.
    """
    prompt = f"Solve the following math problem step by step.\n\nProblem: {problem}\n\nSolution:"
    if is_r1_distill:
        prompt += " <think>"
    return prompt


def generate_cot(
    model_name: str,
    output_path: str,
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    split: str = "test",
    temperature: float = 0.6,
    max_new_tokens: int = 2048,
    max_samples: int | None = None,
) -> None:
    """Generate chain-of-thought reasoning and save as JSONL.

    Args:
        model_name: HuggingFace model name or path.
        output_path: Path to save the output JSONL file.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration name.
        split: Dataset split to use.
        temperature: Sampling temperature.
        max_new_tokens: Maximum new tokens to generate.
        max_samples: If set, limit the number of samples processed.
    """
    is_r1_distill = "r1-distill" in model_name.lower() or "r1_distill" in model_name.lower()

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

    print(f"Loading dataset: {dataset_name}/{dataset_config} [{split}]")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f_out:
        for idx, example in enumerate(tqdm(dataset, desc="Generating CoT")):
            problem_text = example["question"]
            ground_truth = extract_ground_truth(example["answer"])

            prompt = build_prompt(problem_text, is_r1_distill)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_cot = tokenizer.decode(generated_ids, skip_special_tokens=True)

            record = {
                "problem_id": str(idx),
                "problem_text": problem_text,
                "ground_truth_answer": ground_truth,
                "generated_cot": generated_cot,
                "model_name": model_name,
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"Saved {idx + 1} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chain-of-thought reasoning for GSM8K")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    generate_cot(
        model_name=args.model_name,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
