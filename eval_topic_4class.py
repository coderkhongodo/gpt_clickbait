import os
import json
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "gpt-oss-20b-qlora-topic/best")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
TEST_FILE = os.environ.get("TEST_FILE", os.path.join(DATA_DIR, "test_instruction.jsonl"))
NUM = int(os.environ.get("NUM", 1000))
ALLOWED_LABELS = os.environ.get("ALLOWED_LABELS", "0123")
ALLOWED_SET = set(ALLOWED_LABELS)


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
		if ch in ALLOWED_SET:
			return ch
	for ch in ALLOWED_LABELS:
		if f" {ch}" in text:
			return ch
	return ""


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
	class_counts = {c: 0 for c in ALLOWED_LABELS}
	num_match = 0
	y_true = []
	y_pred = []

	for ex in tqdm(test_records, desc="eval-topic", leave=True):
		inst = ex.get("instruction", "").strip()
		inp = ex.get("input", "").strip()
		label = ex.get("output", "").strip()
		# Enforce 4-class answer
		prompt = f"{inst} {inp} Trả lời chỉ 0, 1, 2 hoặc 3:"

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
		if pred in class_counts:
			class_counts[pred] += 1
		if pred == label:
			num_match += 1
		if label in ALLOWED_SET and pred in ALLOWED_SET:
			y_true.append(label)
			y_pred.append(pred)

	print("Pred counts -> " + ", ".join([f"{c}: {class_counts[c]}" for c in ALLOWED_LABELS]))
	print(f"Accuracy on first {len(test_records)}: {num_match/len(test_records):.4f}")
	if y_true:
		acc = accuracy_score(y_true, y_pred)
		prec = precision_score(y_true, y_pred, labels=list(ALLOWED_LABELS), average='macro', zero_division=0)
		rec = recall_score(y_true, y_pred, labels=list(ALLOWED_LABELS), average='macro', zero_division=0)
		f1 = f1_score(y_true, y_pred, labels=list(ALLOWED_LABELS), average='macro', zero_division=0)
		print(
			"Test metrics: "
			f"acc={acc:.4f} f1_macro={f1:.4f} prec_macro={prec:.4f} recall_macro={rec:.4f}"
		)
		print("\nClassification report:\n")
		print(classification_report(y_true, y_pred, labels=list(ALLOWED_LABELS), digits=4, zero_division=0))
		print("Confusion matrix (rows=true, cols=pred):")
		print(confusion_matrix(y_true, y_pred, labels=list(ALLOWED_LABELS)))


if __name__ == "__main__":
	main()


