import os
import json
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "gpt-oss-20b-qlora-finetune")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
TEST_FILE = os.environ.get("TEST_FILE", os.path.join(DATA_DIR, "test_instruction.jsonl"))
NUM = int(os.environ.get("NUM", 100))


def read_jsonl(path: str) -> List[dict]:
	recs = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			recs.append(json.loads(line))
	return recs


def extract_label(text: str) -> str:
	text = text.strip().replace('"', '').replace("'", "").strip()
	for ch in text:
		if ch in ("0", "1"):
			return ch
	if " 0" in text:
		return "0"
	if " 1" in text:
		return "1"
	return ""


def compute_binary_metrics(y_true, y_pred):
	n = len(y_true)
	tp = sum(1 for t, p in zip(y_true, y_pred) if t == "1" and p == "1")
	tn = sum(1 for t, p in zip(y_true, y_pred) if t == "0" and p == "0")
	fp = sum(1 for t, p in zip(y_true, y_pred) if t == "0" and p == "1")
	fn = sum(1 for t, p in zip(y_true, y_pred) if t == "1" and p == "0")

	prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0

	prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
	rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
	f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0

	macro_f1 = (f1_pos + f1_neg) / 2.0
	macro_prec = (prec_pos + prec_neg) / 2.0
	macro_rec = (rec_pos + rec_neg) / 2.0
	acc = (tp + tn) / n if n > 0 else 0.0

	return {
		"accuracy": acc,
		"precision_macro": macro_prec,
		"recall_macro": macro_rec,
		"f1_macro": macro_f1,
		"tp": tp,
		"tn": tn,
		"fp": fp,
		"fn": fn,
	}


def main():
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	cfg = AutoConfig.from_pretrained(MODEL_ID)
	has_prequant = hasattr(cfg, "quantization_config") and cfg.quantization_config is not None

	if has_prequant:
		base_model = AutoModelForCausalLM.from_pretrained(
			MODEL_ID,
			device_map="auto",
			attn_implementation="eager",
			use_cache=True,
			low_cpu_mem_usage=True,
		)
	else:
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_compute_dtype=torch.bfloat16,
		)
		base_model = AutoModelForCausalLM.from_pretrained(
			MODEL_ID,
			device_map="auto",
			attn_implementation="eager",
			use_cache=True,
			quantization_config=quant_config,
			low_cpu_mem_usage=True,
		)

	model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
	model.eval()

	test_records = read_jsonl(TEST_FILE)[:NUM]
	num_0 = 0
	num_1 = 0
	num_match = 0
	y_true = []
	y_pred = []

	for ex in tqdm(test_records, desc="eval-100", leave=True):
		inst = ex.get("instruction", "").strip()
		inp = ex.get("input", "").strip()
		label = ex.get("output", "").strip()
		prompt = f"{inst} {inp} Trả lời chỉ 0 hoặc 1:"

		inputs = tokenizer([prompt], return_tensors="pt")
		inputs = {k: v.to(model.device) for k, v in inputs.items()}

		with torch.no_grad():
			out = model.generate(
				input_ids=inputs["input_ids"],
				attention_mask=inputs["attention_mask"],
				max_new_tokens=4,
				do_sample=False,
				eos_token_id=tokenizer.eos_token_id,
				pad_token_id=tokenizer.eos_token_id,
				use_cache=True,
			)

		new_tokens = out[0][inputs["input_ids"].shape[1]:]
		gen = tokenizer.decode(new_tokens, skip_special_tokens=True)
		pred = extract_label(gen)
		if pred == "0":
			num_0 += 1
		elif pred == "1":
			num_1 += 1
		if pred == label:
			num_match += 1
		if label in ("0", "1") and pred in ("0", "1"):
			y_true.append(label)
			y_pred.append(pred)

	print(f"Pred counts -> 0: {num_0}, 1: {num_1}")
	print(f"Accuracy on first {len(test_records)}: {num_match/len(test_records):.4f}")
	if y_true:
		metrics = compute_binary_metrics(y_true, y_pred)
		print(
			"Test metrics: "
			f"acc={metrics['accuracy']:.4f} "
			f"f1_macro={metrics['f1_macro']:.4f} "
			f"prec_macro={metrics['precision_macro']:.4f} "
			f"recall_macro={metrics['recall_macro']:.4f}"
		)
		# Classification report per class for clarity
		labels_sorted = sorted(list(set(y_true + y_pred)))
		print("\nClassification report:\n")
		print(classification_report(y_true, y_pred, labels=labels_sorted, digits=4))
		# Optional: Confusion matrix
		print("Confusion matrix (rows=true, cols=pred):")
		print(confusion_matrix(y_true, y_pred, labels=labels_sorted))


if __name__ == "__main__":
	main()


