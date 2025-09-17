import os
import json
import csv
import argparse
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-adapted GPT-OSS model on instruction JSONL test set")
    parser.add_argument("--model_id", type=str, default=os.environ.get("MODEL_ID", "openai/gpt-oss-20b"))
    parser.add_argument("--adapter_dir", type=str, default=os.environ.get("ADAPTER_DIR", "models/gpt-oss-20b-qlora-sent-3cls-test/best"))
    parser.add_argument("--test_file", type=str, default=os.environ.get("TEST_FILE", os.path.join("data_processed", "jsonl_text_vsfc_sentiment", "test_instruction.jsonl")))
    parser.add_argument("--output_csv", type=str, default=os.environ.get("OUTPUT_CSV", "evaluation_results.csv"))
    parser.add_argument("--summary_json", type=str, default=os.environ.get("SUMMARY_JSON", "evaluation_summary.json"))
    parser.add_argument("--report_txt", type=str, default=os.environ.get("REPORT_TXT", "classification_report.txt"))
    parser.add_argument("--cm_csv", type=str, default=os.environ.get("CM_CSV", "confusion_matrix.csv"))
    parser.add_argument("--allowed_labels", type=str, default=os.environ.get("ALLOWED_LABELS", "012"), help="Set of allowed labels, e.g. '01' or '0123'")
    parser.add_argument("--max_samples", type=int, default=int(os.environ.get("MAX_SAMPLES", 0)), help="Limit number of evaluated samples (0 = all)")
    return parser.parse_args()


def load_jsonl_as_list(path: str) -> List[dict]:
    """Load JSONL file as list of dictionaries"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_model_and_tokenizer(model_id: str, adapter_dir: str):
    """Load the fine-tuned model and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    # Load base model using same approach as training (detect pre-quantization)
    cfg = AutoConfig.from_pretrained(model_id)
    has_prequant = hasattr(cfg, "quantization_config") and cfg.quantization_config is not None

    if has_prequant:
        # Model already carries a quantization config (e.g., MXFP4). Load as-is.
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="eager",
            use_cache=True,
            low_cpu_mem_usage=True,
        )
    else:
        # Fallback to 4-bit bnb quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="eager",
            use_cache=True,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    
    return model, tokenizer


def extract_prediction(generated_text: str, allowed: str) -> str:
    """Extract prediction from generated text constrained to allowed labels (e.g. '012')."""
    generated_text = generated_text.strip()
    cleaned = generated_text.replace('"', '').replace("'", '').strip()
    # Priority: first character that is in allowed
    for ch in cleaned:
        if ch in allowed:
            return ch
    # Secondary: look for space-prefixed token
    for ch in allowed:
        if f" {ch}" in cleaned:
            return ch
    # Fallback: return first char if it matches digit
    return cleaned[:1] if cleaned[:1] in allowed else allowed[0]


def predict_single_example(model, tokenizer, instruction: str, input_text: str, allowed: str) -> str:
    """Generate prediction for a single example"""
    # Tạo prompt theo format instruction
    if input_text:
        prompt = f"{instruction}\n\n{input_text}\n\n"
    else:
        prompt = f"{instruction}\n\n"
    
    # Tokenize input
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with memory optimization
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5,  # Giảm xuống 5 tokens
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            use_cache=False,  # Tắt cache để tiết kiệm bộ nhớ
            output_attentions=False,
            output_hidden_states=False,
        )
    
    # Decode only the new tokens (excluding input prompt)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract prediction
    prediction = extract_prediction(generated_text, allowed)
    
    # Clear cache after each prediction
    torch.cuda.empty_cache()
    
    return prediction


def evaluate_model():
    args = parse_args()
    """Evaluate model on entire test set and save results to CSV"""
    print("Loading test data...")
    test_records = load_jsonl_as_list(args.test_file)
    if args.max_samples and args.max_samples > 0:
        test_records = test_records[:args.max_samples]
    print(f"Loaded {len(test_records)} test examples")
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.adapter_dir)
    
    # Prepare results storage
    results = []
    true_labels = []
    predicted_labels = []
    
    print("Starting evaluation...")
    for i, example in enumerate(test_records):
        if i % 10 == 0:  # Clear cache more frequently
            torch.cuda.empty_cache()
            print(f"Processing example {i+1}/{len(test_records)}")
        
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        true_output = example.get("output", "")
        
        # Extract true label (remove </s> if present)
        true_label = true_output.replace("</s>", "").strip()
        
        # Generate prediction
        try:
            prediction = predict_single_example(model, tokenizer, instruction, input_text, args.allowed_labels)
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            prediction = "error"
            # Clear cache on error
            torch.cuda.empty_cache()
        
        # Store results
        result = {
            "example_id": i,
            "instruction": instruction,
            "input": input_text,
            "true_label": true_label,
            "predicted_label": prediction,
            "correct": true_label == prediction
        }
        results.append(result)
        
        # Store for metrics calculation
        if prediction != "error":
            true_labels.append(true_label)
            predicted_labels.append(prediction)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    if true_labels and predicted_labels:
        labels_sorted = sorted(list(set(list(args.allowed_labels))))
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Calculate different F1 scores
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels_sorted, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels_sorted, average='weighted', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels_sorted, average='micro', zero_division=0
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\n=== F1 SCORES ===")
        print(f"F1-Score (macro): {f1_macro:.4f}")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
        print(f"F1-Score (micro): {f1_micro:.4f}")
        print(f"\n=== PRECISION ===")
        print(f"Precision (macro): {precision_macro:.4f}")
        print(f"Precision (weighted): {precision_weighted:.4f}")
        print(f"Precision (micro): {precision_micro:.4f}")
        print(f"\n=== RECALL ===")
        print(f"Recall (macro): {recall_macro:.4f}")
        print(f"Recall (weighted): {recall_weighted:.4f}")
        print(f"Recall (micro): {recall_micro:.4f}")
        # Detailed classification report
        print("\nDetailed Classification Report:")
        report = classification_report(true_labels, predicted_labels, labels=labels_sorted, digits=4, zero_division=0)
        print(report)
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels_sorted)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        # Save report and confusion matrix
        with open(args.report_txt, 'w', encoding='utf-8') as f:
            f.write(report)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_sorted], columns=[f"pred_{l}" for l in labels_sorted])
        cm_df.to_csv(args.cm_csv, index=True, encoding='utf-8')
    
    # Save results to CSV
    print(f"\nSaving results to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['example_id', 'instruction', 'input', 'true_label', 'predicted_label', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Create summary
    total_examples = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    error_predictions = sum(1 for r in results if r['predicted_label'] == 'error')
    
    print(f"\nSummary:")
    print(f"Total examples: {total_examples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Error predictions: {error_predictions}")
    print(f"Success rate: {correct_predictions/total_examples:.4f}")
    
    # Save summary metrics
    summary = {
        'total_examples': total_examples,
        'correct_predictions': correct_predictions,
        'error_predictions': error_predictions,
        'success_rate': correct_predictions/total_examples,
        'accuracy': accuracy if 'accuracy' in locals() else 0,
        'precision_macro': precision_macro if 'precision_macro' in locals() else 0,
        'precision_weighted': precision_weighted if 'precision_weighted' in locals() else 0,
        'precision_micro': precision_micro if 'precision_micro' in locals() else 0,
        'recall_macro': recall_macro if 'recall_macro' in locals() else 0,
        'recall_weighted': recall_weighted if 'recall_weighted' in locals() else 0,
        'recall_micro': recall_micro if 'recall_micro' in locals() else 0,
        'f1_macro': f1_macro if 'f1_macro' in locals() else 0,
        'f1_weighted': f1_weighted if 'f1_weighted' in locals() else 0,
        'f1_micro': f1_micro if 'f1_micro' in locals() else 0
    }
    
    with open(args.summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_csv}")
    print(f"Summary saved to {args.summary_json}")


if __name__ == "__main__":
    evaluate_model()
