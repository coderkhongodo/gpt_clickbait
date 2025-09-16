import os
import json
from typing import List, Tuple


DATASET_DIR = os.environ.get("VSFC_DIR", os.path.join("uit-vsfc"))
SPLITS = ["train", "dev", "test"]

# Output directory follows existing project convention
OUTPUT_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))


def read_split(split_dir: str) -> Tuple[List[str], List[str]]:

	texts_path = os.path.join(split_dir, "sents.txt")
	labels_path = os.path.join(split_dir, "sentiments.txt")

	with open(texts_path, "r", encoding="utf-8") as f_txt:
		texts = [line.rstrip("\n") for line in f_txt]

	with open(labels_path, "r", encoding="utf-8") as f_lab:
		labels = [line.strip() for line in f_lab]

	if len(texts) != len(labels):
		raise ValueError(f"Mismatched lines: {texts_path} ({len(texts)}) vs {labels_path} ({len(labels)})")

	return texts, labels


def map_label_to_binary(label: str) -> str:
	# VSFC: 0 = negative, 1 = neutral, 2 = positive
	# We drop 1 (neutral). Map: 0 -> "0", 2 -> "1"
	if label == "0":
		return "0"
	if label == "2":
		return "1"
	return ""  # neutral or unknown -> filtered out


def ensure_output_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def write_jsonl(records: List[dict], out_path: str) -> None:
	with open(out_path, "w", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def convert_split(split: str) -> str:
	split_dir = os.path.join(DATASET_DIR, split)
	texts, labels = read_split(split_dir)

	records: List[dict] = []
	inst = "Hãy phân loại cảm xúc câu sau là 0 (tiêu cực) hoặc 1 (tích cực):"

	for text, raw_label in zip(texts, labels):
		bin_label = map_label_to_binary(raw_label)
		if not bin_label:
			continue
		records.append({
			"instruction": inst,
			"input": text,
			"output": bin_label
		})

	ensure_output_dir(OUTPUT_DIR)
	out_name = {
		"train": "train_instruction.jsonl",
		"dev": "val_instruction.jsonl",
		"test": "test_instruction.jsonl",
	}[split]

	out_path = os.path.join(OUTPUT_DIR, out_name)
	write_jsonl(records, out_path)
	print(f"Wrote {len(records)} records to {out_path}")
	return out_path


def main():
	for split in SPLITS:
		convert_split(split)


if __name__ == "__main__":
	main()


