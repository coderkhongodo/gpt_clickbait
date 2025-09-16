import os
import json
import random
from typing import List, Dict, Tuple


SOURCE_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
TARGET_DIR = os.environ.get("SMALL_DATA_DIR", os.path.join("jsonl_text_small"))

TRAIN_FILE = os.path.join(SOURCE_DIR, "train_instruction.jsonl")
VAL_FILE = os.path.join(SOURCE_DIR, "val_instruction.jsonl")
TEST_FILE = os.path.join(SOURCE_DIR, "test_instruction.jsonl")


def read_jsonl(path: str) -> List[Dict]:
	with open(path, "r", encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records: List[Dict]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sample_balanced(records: List[Dict], per_class: int) -> List[Dict]:
	neg = [r for r in records if r.get("output") == "0"]
	pos = [r for r in records if r.get("output") == "1"]
	random.shuffle(neg)
	random.shuffle(pos)
	return neg[:per_class] + pos[:per_class]


def build_prompts_from_instruction(records: List[Dict]) -> List[Dict]:
	prompts = []
	for r in records:
		inst = r.get("instruction", "").strip()
		inp = r.get("input", "").strip()
		if inp:
			prompt = f"{inst} {inp}"
		else:
			prompt = inst
		prompts.append({"prompt": prompt})
	return prompts


def main():
	random.seed(42)

	train = read_jsonl(TRAIN_FILE)
	val = read_jsonl(VAL_FILE)
	test = read_jsonl(TEST_FILE)

	train_small = sample_balanced(train, per_class=500)
	val_small = sample_balanced(val, per_class=100)
	# Build a small prompt-only test set for the inference script (5 mixed samples)
	test_small_candidates = sample_balanced(test, per_class=5)
	test_small_prompts = build_prompts_from_instruction(test_small_candidates)

	write_jsonl(os.path.join(TARGET_DIR, "train_instruction.jsonl"), train_small)
	write_jsonl(os.path.join(TARGET_DIR, "val_instruction.jsonl"), val_small)
	write_jsonl(os.path.join(TARGET_DIR, "test.jsonl"), test_small_prompts)

	print(f"Wrote {len(train_small)} train, {len(val_small)} val, {len(test_small_prompts)} test prompts to {TARGET_DIR}")


if __name__ == "__main__":
	main()


