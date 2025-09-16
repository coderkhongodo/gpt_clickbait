import os
import json


DATASET_DIR = os.environ.get("VSFC_DIR", os.path.join("data", "uit-vsfc"))
SPLITS = ["train", "dev", "test"]
OUTPUT_DIR = os.environ.get("DATA_DIR", os.path.join("data_processed", "jsonl_text_vsfc_topic"))


def read_split(split_dir: str):
	texts_path = os.path.join(split_dir, "sents.txt")
	labels_path = os.path.join(split_dir, "topics.txt")

	with open(texts_path, "r", encoding="utf-8") as f_txt:
		texts = [line.rstrip("\n") for line in f_txt]

	with open(labels_path, "r", encoding="utf-8") as f_lab:
		labels = [line.strip() for line in f_lab]

	if len(texts) != len(labels):
		raise ValueError(f"Mismatched lines: {texts_path} vs {labels_path}")

	return texts, labels


def ensure_output_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def write_jsonl(records, out_path: str) -> None:
	with open(out_path, "w", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def convert_split(split: str) -> str:
	split_dir = os.path.join(DATASET_DIR, split)
	texts, labels = read_split(split_dir)

	records = []
	inst = (
		"Bạn là bộ phân loại chủ đề (topic).\n"
		"Quy ước lớp:\n"
		"- 0 = giảng viên\n"
		"- 1 = chương trình đào tạo\n"
		"- 2 = cơ sở vật chất\n"
		"- 3 = khác\n\n"
		"Chỉ trả đúng MỘT ký tự trong {0,1,2,3}. Không thêm chữ nào khác.\n\n"
		"Câu cần phân loại:"
	)

	for text, label in zip(texts, labels):
		label = label.strip()
		if label not in {"0", "1", "2", "3"}:
			continue
		records.append({
			"instruction": inst,
			"input": f"{text}\nĐáp án:",
			"output": label
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


