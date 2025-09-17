import os
import json
from typing import List, Tuple


DATASET_DIR = os.environ.get("VSFC_DIR", os.path.join("data", "uit-vsfc"))
SPLITS = ["train", "dev", "test"]

# Output directory for UIT-VSFC sentiment (2-class) instructions
OUTPUT_DIR = os.environ.get("DATA_DIR", os.path.join("data_processed", "jsonl_text_vsfc_sentiment_2cls"))


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


def map_label_2class(label: str) -> str:
	# Original: 0=neg, 1=neutral, 2=pos
	# New 2-class mapping: drop neutral; 0=neg, 1=pos
	label = label.strip()
	if label == "0":
		return "0"
	if label == "2":
		return "1"
	return ""  # ignore other labels (e.g., neutral "1")


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
	inst = (
		"Bạn là bộ phân loại cảm xúc 2 lớp.\n"
		"Quy ước:\n"
		"- 0 = negative: có ý chê, phàn nàn, bất mãn.\n"
		"- 1 = positive: khen ngợi, hài lòng, đánh giá tích cực.\n\n"
		"Chỉ trả đúng MỘT ký tự trong {0,1}. Không thêm chữ nào khác.\n\n"
		"Câu cần phân loại:"
	)

	for text, raw_label in zip(texts, labels):
		mapped = map_label_2class(raw_label)
		if mapped not in {"0", "1"}:
			continue
		records.append({
			"instruction": inst,
			"input": f"{text}\nĐáp án:",
			"output": mapped
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


