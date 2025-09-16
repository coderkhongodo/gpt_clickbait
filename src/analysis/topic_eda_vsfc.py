import os
import json
import argparse
from typing import List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SPLITS = ["train", "dev", "test"]


def resolve_vsfc_dir(vsfc_dir: str) -> str:
	if os.path.isabs(vsfc_dir) and os.path.exists(vsfc_dir):
		return vsfc_dir
	cwd = os.getcwd()
	candidate = cwd
	while True:
		cand_path = os.path.join(candidate, vsfc_dir)
		if all(os.path.exists(os.path.join(cand_path, sp, "sents.txt")) and os.path.exists(os.path.join(cand_path, sp, "topics.txt")) for sp in SPLITS):
			return cand_path
		parent = os.path.dirname(candidate)
		if parent == candidate:
			break
		candidate = parent
	return vsfc_dir


def read_split(split_dir: str) -> Tuple[List[str], List[str]]:
	texts_path = os.path.join(split_dir, "sents.txt")
	labels_path = os.path.join(split_dir, "topics.txt")
	with open(texts_path, "r", encoding="utf-8") as f_txt:
		texts = [line.rstrip("\n") for line in f_txt]
	with open(labels_path, "r", encoding="utf-8") as f_lab:
		labels = [line.strip() for line in f_lab]
	if len(texts) != len(labels):
		raise ValueError(f"Mismatched lines: {texts_path} vs {labels_path}")
	return texts, labels


def build_dataframe(vsfc_dir: str) -> pd.DataFrame:
	rows = []
	for split in SPLITS:
		split_dir = os.path.join(vsfc_dir, split)
		texts, labels = read_split(split_dir)
		for t, lab in zip(texts, labels):
			if lab not in {"0", "1", "2", "3"}:
				continue
			rows.append({"split": split, "text": t, "label": lab})
	df = pd.DataFrame(rows)
	df["len"] = df["text"].str.len()
	return df


def save_plots(df: pd.DataFrame, out_dir: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	plt.figure(figsize=(10,4))
	sns.countplot(data=df, x="label", hue="split")
	plt.title("Topic label distribution by split (0,1,2,3)")
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "topic_label_distribution_by_split.png"), dpi=150)
	plt.close()

	plt.figure(figsize=(10,4))
	sns.kdeplot(data=df, x="len", hue="split", common_norm=False, fill=True, alpha=0.3)
	plt.title("Sentence length (chars) by split")
	plt.xlim(0, df["len"].quantile(0.99))
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "topic_length_kde_by_split.png"), dpi=150)
	plt.close()

	plt.figure(figsize=(8,4))
	sns.boxplot(data=df, x="label", y="len")
	plt.title("Length by topic label")
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "topic_length_box_by_label.png"), dpi=150)
	plt.close()


def save_preview_jsonl(df: pd.DataFrame, out_path: str, num: int = 200) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
	preview = (
		df[df["split"] == "train"][ ["text", "label"] ]
		.head(num)
		.assign(instruction=inst)
		.rename(columns={"text": "input", "label": "output"})
	)
	# Append "Đáp án:" marker to input
	preview["input"] = preview["input"].astype(str) + "\nĐáp án:"
	with open(out_path, "w", encoding="utf-8") as f:
		for _, r in preview.iterrows():
			f.write(json.dumps({k: r[k] for k in ["instruction", "input", "output"]}, ensure_ascii=False) + "\n")


def main():
	parser = argparse.ArgumentParser(description="UIT-VSFC Topic EDA (4 classes)")
	parser.add_argument("--vsfc_dir", type=str, default=os.path.join("data", "uit-vsfc"))
	parser.add_argument("--out_dir", type=str, default=os.path.join("analysis_outputs", "topic"))
	parser.add_argument("--save_plots", action="store_true")
	parser.add_argument("--save_preview", action="store_true")
	parser.add_argument("--preview_path", type=str, default=os.path.join("data_processed", "jsonl_text_topic_preview.jsonl"))
	args = parser.parse_args()

	vsfc_dir = resolve_vsfc_dir(args.vsfc_dir)
	print(f"CWD={os.getcwd()}")
	print(f"Using VSFC_DIR={vsfc_dir}")

	df = build_dataframe(vsfc_dir)
	print("Label distribution by split (0,1,2,3):")
	print(df.groupby(["split", "label"]).size().reset_index(name="count").sort_values(["split", "label"]).to_string(index=False))
	print()
	print("Total samples:", len(df))
	print()
	print("Length statistics by split:")
	print(df.groupby("split")["len"].describe().to_string())
	print()
	print("Length statistics by label:")
	print(df.groupby("label")["len"].describe().to_string())

	if args.save_plots:
		save_plots(df, args.out_dir)
		print(f"Saved plots to {args.out_dir}")

	if args.save_preview:
		save_preview_jsonl(df, args.preview_path)
		print(f"Wrote preview JSONL to {args.preview_path}")

	# Print a few examples per label
	for lbl in ["0", "1", "2", "3"]:
		sub = df[df["label"] == lbl].sample(n=min(5, len(df[df["label"] == lbl])), random_state=42)
		print(f"\n=== Examples label {lbl} ===")
		for _, row in sub.iterrows():
			print(f"[{row['split']}] {row['text']}")


if __name__ == "__main__":
	main()


