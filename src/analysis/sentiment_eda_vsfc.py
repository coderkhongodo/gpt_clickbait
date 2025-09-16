import os
import json
import argparse
from typing import List, Tuple, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SPLITS = ["train", "dev", "test"]


def resolve_vsfc_dir(vsfc_dir: str) -> str:
	"""Resolve the UIT-VSFC directory.

	- If absolute and exists, return as-is
	- Otherwise, walk up from CWD to find a folder named vsfc_dir that contains
	  expected files for all SPLITS
	"""
	if os.path.isabs(vsfc_dir) and os.path.exists(vsfc_dir):
		return vsfc_dir

	cwd = os.getcwd()
	candidate = cwd
	while True:
		cand_path = os.path.join(candidate, vsfc_dir)
		if all(
			os.path.exists(os.path.join(cand_path, sp, "sents.txt")) and
			os.path.exists(os.path.join(cand_path, sp, "sentiments.txt"))
			for sp in SPLITS
		):
			return cand_path
		parent = os.path.dirname(candidate)
		if parent == candidate:
			break
		candidate = parent
	return vsfc_dir


def read_split(split_dir: str) -> Tuple[List[str], List[str]]:
	texts_path = os.path.join(split_dir, "sents.txt")
	labels_path = os.path.join(split_dir, "sentiments.txt")
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
			rows.append({"split": split, "text": t, "label": lab})
	df = pd.DataFrame(rows)
	df["len"] = df["text"].str.len()
	return df


def print_label_distribution(df: pd.DataFrame) -> None:
	print("Label distribution by split (0=neg,1=neutral,2=pos):")
	counts = (
		df.groupby(["split", "label"]).size().reset_index(name="count")
		.sort_values(["split", "label"]).reset_index(drop=True)
	)
	print(counts.to_string(index=False))
	print()
	print("Total samples:", len(df))


def print_length_stats(df: pd.DataFrame) -> None:
	print("Length statistics by split:")
	print(df.groupby("split")["len"].describe().to_string())
	print()
	print("Length statistics by label:")
	print(df.groupby("label")["len"].describe().to_string())


def save_plots(df: pd.DataFrame, out_dir: str) -> None:
	os.makedirs(out_dir, exist_ok=True)

	plt.figure(figsize=(10, 4))
	sns.countplot(data=df, x="label", hue="split")
	plt.title("Label distribution by split (0=neg,1=neutral,2=pos)")
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "label_distribution_by_split.png"), dpi=150)
	plt.close()

	plt.figure(figsize=(10, 4))
	sns.kdeplot(data=df, x="len", hue="split", common_norm=False, fill=True, alpha=0.3)
	plt.title("Sentence length (chars) by split")
	plt.xlim(0, df["len"].quantile(0.99))
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "length_kde_by_split.png"), dpi=150)
	plt.close()

	plt.figure(figsize=(8, 4))
	sns.boxplot(data=df, x="label", y="len")
	plt.title("Length by label (0=neg,1=neutral,2=pos)")
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "length_box_by_label.png"), dpi=150)
	plt.close()


def save_preview_jsonl(df: pd.DataFrame, out_path: str, num: int = 200) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	preview = (
		df[df["split"] == "train"][ ["text", "label"] ]
		.head(num)
		.assign(
			instruction=(
				"Hãy phân loại cảm xúc câu sau theo 3 lớp: 0 (tiêu cực), 1 (trung lập), 2 (tích cực). "
				"Trả lời chỉ 0, 1 hoặc 2:"
			)
		)
		.rename(columns={"text": "input", "label": "output"})
	)
	with open(out_path, "w", encoding="utf-8") as f:
		for _, r in preview.iterrows():
			f.write(json.dumps({k: r[k] for k in ["instruction", "input", "output"]}, ensure_ascii=False) + "\n")


def sample_examples(df: pd.DataFrame, per_class: int = 5) -> Dict[str, List[str]]:
	examples: Dict[str, List[str]] = {"0": [], "1": [], "2": []}
	for label in ["0", "1", "2"]:
		sub = df[df["label"] == label].sample(
			n=min(per_class, len(df[df["label"] == label])),
			random_state=42,
		)
		for _, row in sub.iterrows():
			examples[label].append(f"[{row['split']}] {row['text']}")
	return examples


def main():
	parser = argparse.ArgumentParser(description="UIT-VSFC Sentiment EDA (3 classes)")
	parser.add_argument(
		"--vsfc_dir",
		type=str,
		default=os.environ.get("VSFC_DIR", os.path.join("data", "uit-vsfc")),
		help="Path to uit-vsfc directory (containing train/dev/test)",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default=os.path.join("analysis_outputs", "sentiment"),
		help="Directory to save figures and preview JSONL",
	)
	parser.add_argument(
		"--save_plots",
		action="store_true",
		help="If set, save distribution and length plots",
	)
	parser.add_argument(
		"--save_preview",
		action="store_true",
		help="If set, write a small 3-class instruction JSONL preview",
	)
	parser.add_argument(
		"--preview_path",
		type=str,
		default=os.path.join("data_processed", "jsonl_text", "train_instruction_preview_3class.jsonl"),
		help="Where to write the preview JSONL if --save_preview",
	)
	args = parser.parse_args()

	vsfc_dir = resolve_vsfc_dir(args.vsfc_dir)
	print(f"CWD={os.getcwd()}")
	print(f"Using VSFC_DIR={vsfc_dir}")

	# Validate
	missing = [
		sp
		for sp in SPLITS
		if not (
			os.path.exists(os.path.join(vsfc_dir, sp, "sents.txt"))
			and os.path.exists(os.path.join(vsfc_dir, sp, "sentiments.txt"))
		)
	]
	if missing:
		raise FileNotFoundError(
			"UIT-VSFC not found or incomplete (missing sents.txt/sentiments.txt). "
			"Set VSFC_DIR env or --vsfc_dir to the correct path."
		)

	df = build_dataframe(vsfc_dir)
	print_label_distribution(df)
	print_length_stats(df)

	if args.save_plots:
		save_plots(df, args.out_dir)
		print(f"Saved plots to {args.out_dir}")

	if args.save_preview:
		save_preview_jsonl(df, args.preview_path)
		print(f"Wrote preview JSONL to {args.preview_path}")

	# Print a few examples
	examples = sample_examples(df, per_class=5)
	for lbl, items in examples.items():
		print(f"\n=== Examples label {lbl} ===")
		for line in items:
			print(line)


if __name__ == "__main__":
	main()


