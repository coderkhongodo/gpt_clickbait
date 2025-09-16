import os
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import AutoConfig
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from tqdm import tqdm


# ------------------------------
# Config
# ------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "gpt-oss-20b-qlora-finetune")

# JSONL files must exist inside DATA_DIR - sử dụng format instruction mới
TRAIN_FILE = os.path.join(DATA_DIR, "train_instruction.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val_instruction.jsonl")


def assert_files():
    for p in [TRAIN_FILE, VAL_FILE]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")


def load_jsonl_as_hf_dataset(path: str):
    # Use datasets' json loader which supports JSON Lines
    return load_dataset("json", data_files=path, split="train")


@dataclass
class FormattingConfig:
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    add_eos: bool = False  # Không cần thêm EOS vì output đã có </s>


def build_text(example: Dict, tok, cfg: FormattingConfig) -> str:
    # Format mới: instruction + input + output (đã có </s)
    instruction = example.get(cfg.instruction_key, "")
    input_text = example.get(cfg.input_key, "")
    output = example.get(cfg.output_key, "")
    
    # Tạo text theo format: instruction + input + output
    if input_text:
        text = f"{instruction}\n\n{input_text}\n\n{output}"
    else:
        text = f"{instruction}\n\n{output}"
    
    return text


def prepare_dataset(ds, tok, cfg: FormattingConfig):
    def _map_fn(batch):
        instructions = batch[cfg.instruction_key]
        inputs = batch[cfg.input_key]
        outputs = batch[cfg.output_key]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            instruction = instruction or ""
            input_text = input_text or ""
            output = output or ""
            
            # Tạo text theo format mới
            if input_text:
                text = f"{instruction}\n\n{input_text}\n\n{output}"
            else:
                text = f"{instruction}\n\n{output}"
            
            texts.append(text)
        
        return {"text": texts}

    cols = ds.column_names
    # Kiểm tra xem có đủ các trường cần thiết không
    required_cols = [cfg.instruction_key, cfg.output_key]
    missing_cols = [col for col in required_cols if col not in cols]
    
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}. Found: {cols}"
        )
    
    ds = ds.map(_map_fn, batched=True, remove_columns=cols)
    return ds


def get_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def format_text_field(example: Dict) -> str:
    # SFTTrainer (this TRL version) expects a formatting_func to return a string
    # We already mapped dataset to have a single 'text' field
    return example["text"]


def get_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_model(model_id: str):
    cfg = AutoConfig.from_pretrained(model_id)

    has_prequant = hasattr(cfg, "quantization_config") and cfg.quantization_config is not None

    if has_prequant:
        # Model already carries a quantization config (e.g., MXFP4). Load as-is.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="eager",
            use_cache=False,
            low_cpu_mem_usage=True,
        )
    else:
        # Fallback to 4-bit bnb quantization
        quant_config = get_4bit_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation="eager",
            use_cache=False,
            low_cpu_mem_usage=True,
        )

    return model


def attach_lora(base_model):
    peft_config = LoraConfig(
        r=32,  # Tăng từ 16 lên 32 để có capacity cao hơn
        lora_alpha=64,  # Tăng từ 32 lên 64
        lora_dropout=0.1,  # Tăng dropout để tránh overfitting
        bias="none",
        target_modules="all-linear",
        # Thêm các tham số mới
        inference_mode=False,
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model


def main():
    assert_files()

    tokenizer = get_tokenizer(MODEL_ID)

    # Load datasets
    train_ds = load_jsonl_as_hf_dataset(TRAIN_FILE)
    val_ds = load_jsonl_as_hf_dataset(VAL_FILE)

    # Prepare into single text field for SFT
    fmt_cfg = FormattingConfig()
    train_ds = prepare_dataset(train_ds, tokenizer, fmt_cfg)
    val_ds = prepare_dataset(val_ds, tokenizer, fmt_cfg)

    model = get_model(MODEL_ID)
    model = attach_lora(model)

    # Hyperparameters tối ưu cho format instruction mới
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", 1)),
        per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH_SIZE", 1)),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", 16)),  # Giảm từ 32 xuống 16
        learning_rate=float(os.environ.get("LR", 5e-4)),  # Tăng từ 2e-4 lên 5e-4
        num_train_epochs=float(os.environ.get("EPOCHS", 5)),  # Tăng từ 1 lên 5 epochs
        logging_steps=int(os.environ.get("LOG_STEPS", 10)),
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        warmup_ratio=0.1,  # Tăng warmup ratio
        lr_scheduler_type="cosine",
        report_to=os.environ.get("REPORT_TO", "none"),
        optim=os.environ.get("OPTIM", "paged_adamw_8bit"),
        packing=True,
        # Thêm các tham số mới để tối ưu
        dataloader_pin_memory=False,  # Tiết kiệm memory
        remove_unused_columns=False,  # Giữ columns để debug
        # Đánh giá/lưu được xử lý bởi callback tùy biến (EpochEvalCallback)
    )

    def _parse_text_to_prompt_and_label(text: str) -> (str, str):
        # text format: instruction + \n\n + optional input + \n\n + output
        parts = text.split("\n\n")
        if not parts:
            return text, ""
        label = parts[-1].strip()
        prompt_core = " ".join(p.strip() for p in parts[:-1] if p.strip())
        prompt = f"{prompt_core} Trả lời chỉ 0 hoặc 1:"
        return prompt, label

    def _extract_binary_label(gen_text: str) -> str:
        cleaned = gen_text.strip().replace('"', '').replace("'", "").strip()
        for ch in cleaned:
            if ch in ("0", "1"):
                return ch
        if " 0" in cleaned:
            return "0"
        if " 1" in cleaned:
            return "1"
        return ""

    def _compute_binary_metrics(y_true, y_pred):
        # y_true/y_pred are lists of "0"/"1"
        n = len(y_true)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "1" and p == "1")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "0" and p == "0")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "0" and p == "1")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "1" and p == "0")

        # Per-class precision/recall
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

    class TQDMCallback(TrainerCallback):
        def __init__(self):
            self.pbar = None

        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            total = int(state.max_steps) if state.max_steps is not None else None
            self.pbar = tqdm(total=total, desc="training", leave=True)

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if self.pbar is not None:
                self.pbar.update(1)

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if self.pbar is not None:
                self.pbar.close()

    class EpochEvalCallback(TrainerCallback):
        def __init__(self, val_dataset, tokenizer, output_dir, patience_epochs: int = 2):
            self.val_dataset = val_dataset
            self.tokenizer = tokenizer
            self.output_dir = output_dir
            self.best_f1 = -1.0
            self.no_improve = 0
            self.patience = patience_epochs

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
            model.eval()
            y_true = []
            y_pred = []

            for ex in tqdm(self.val_dataset, desc=f"validate epoch {int(state.epoch) if state.epoch is not None else '?'}", leave=False):
                text = ex["text"]
                prompt, label = _parse_text_to_prompt_and_label(text)
                if label not in ("0", "1"):
                    continue

                inputs = self.tokenizer([prompt], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=4,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )

                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                gen = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred = _extract_binary_label(gen)
                if pred not in ("0", "1"):
                    # fallback: treat as wrong prediction
                    pred = "0"

                y_true.append(label)
                y_pred.append(pred)

            metrics = _compute_binary_metrics(y_true, y_pred)
            tqdm.write(
                f"val epoch={state.epoch:.2f} acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} "
                f"prec_macro={metrics['precision_macro']:.4f} rec_macro={metrics['recall_macro']:.4f}"
            )

            # Save best by macro F1
            if metrics["f1_macro"] > self.best_f1:
                self.best_f1 = metrics["f1_macro"]
                self.no_improve = 0
                save_dir = os.path.join(self.output_dir, "best")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                # tokenizer is unchanged, keep reference outside
                tqdm.write(f"saved new best to {save_dir} (f1_macro={self.best_f1:.4f})")
            else:
                self.no_improve += 1
                tqdm.write(f"no improvement epochs={self.no_improve}/{self.patience}")
                if self.no_improve >= self.patience:
                    control.should_training_stop = True
                    tqdm.write("early stopping triggered")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=format_text_field,
    )

    trainer.add_callback(TQDMCallback())
    trainer.add_callback(EpochEvalCallback(val_ds, tokenizer, OUTPUT_DIR, patience_epochs=int(os.environ.get("PATIENCE_EPOCHS", 2))))

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()


