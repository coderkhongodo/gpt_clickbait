import os
import json
import argparse
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from transformers.generation.logits_process import LogitsProcessor
from peft import PeftModel
from tqdm import tqdm


MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "gpt-oss-20b-qlora-finetune")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
TEST_FILE = os.environ.get("TEST_FILE", os.path.join(DATA_DIR, "test.jsonl"))


def load_jsonl_as_list(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class AllowListLogitsProcessor(LogitsProcessor):
    def __init__(self, allow_token_ids):
        self.allow = set(allow_token_ids)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        indices = torch.tensor(list(self.allow), device=scores.device)
        mask.index_fill_(1, indices, 0.0)
        return scores + mask


def parse_args():
    p = argparse.ArgumentParser(description="Inference with optional constrained decoding")
    p.add_argument("--allowed_labels", type=str, default=os.environ.get("ALLOWED_LABELS", "01"))
    p.add_argument("--constrained", action="store_true", help="Enable constrained decoding to allowed labels")
    p.add_argument("--num_samples", type=int, default=int(os.environ.get("NUM_SAMPLES", 5)))
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model using same approach as training (detect pre-quantization)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    has_prequant = hasattr(cfg, "quantization_config") and cfg.quantization_config is not None

    if has_prequant:
        # Model already carries a quantization config (e.g., MXFP4). Load as-is.
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
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
            MODEL_ID,
            device_map="auto",
            attn_implementation="eager",
            use_cache=True,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    test_records = load_jsonl_as_list(TEST_FILE)
    to_show = args.num_samples
    
    # Debug: check tokenizer EOS token
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print()
    
    for ex in tqdm(test_records[:to_show], desc="inference", leave=True):
        prompt = ex.get("prompt", "")
        # Nắn prompt để mô hình trả lời duy nhất 0 hoặc 1
        prompt = f"{prompt.strip()} Trả lời chỉ 0 hoặc 1:"
        # Don't add EOS to input prompt - let model generate it
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                use_cache=True,
            )
            if args.constrained:
                allow_ids = tokenizer.convert_tokens_to_ids(list(args.allowed_labels))
                processor = AllowListLogitsProcessor(allow_ids)
                out = model.generate(**gen_kwargs, logits_processor=processor)
            else:
                out = model.generate(**gen_kwargs)
        
        # Decode only the new tokens (excluding input prompt)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Post-process: chỉ giữ lại "0" hoặc "1" (chống trường hợp sinh ra kí tự thừa như ")
        cleaned_output = generated_text.strip().replace('"', '').replace("'", "").strip()
        # Ưu tiên tìm số đơn lẻ bất kỳ vị trí
        found = None
        for ch in cleaned_output:
            if ch in ("0", "1"):
                found = ch
                break
        if found is None:
            # Thử tìm pattern có khoảng trắng
            if " 0" in cleaned_output:
                found = "0"
            elif " 1" in cleaned_output:
                found = "1"
        cleaned_output = found if found is not None else cleaned_output[:1]
        
        print("==== Prompt ====")
        print(prompt)
        print("==== Model output ====")
        print(cleaned_output)
        print()


if __name__ == "__main__":
    main()


