import os
import json
import argparse
from typing import List
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-OSS 20B with QLoRA (SFT)")
    # Paths
    parser.add_argument("--model_id", type=str, default=os.environ.get("MODEL_ID", "openai/gpt-oss-20b"))
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", os.path.join("jsonl_text")))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "gpt-oss-20b-qlora-finetune"))
    parser.add_argument("--train_file", type=str, default=os.environ.get("TRAIN_FILE", "train_instruction.jsonl"))
    parser.add_argument("--val_file", type=str, default=os.environ.get("VAL_FILE", "val_instruction.jsonl"))
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 1)))
    parser.add_argument("--eval_batch_size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", 1)))
    parser.add_argument("--grad_accum", type=int, default=int(os.environ.get("GRAD_ACCUM", 16)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", 5e-4)))
    parser.add_argument("--epochs", type=float, default=float(os.environ.get("EPOCHS", 5)))
    parser.add_argument("--log_steps", type=int, default=int(os.environ.get("LOG_STEPS", 10)))
    parser.add_argument("--optim", type=str, default=os.environ.get("OPTIM", "paged_adamw_8bit"))
    parser.add_argument("--report_to", type=str, default=os.environ.get("REPORT_TO", "none"))
    parser.add_argument("--warmup_ratio", type=float, default=float(os.environ.get("WARMUP_RATIO", 0.1)))
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 (default)")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16")
    parser.add_argument("--packing", action="store_true", help="Enable packing (default)")
    parser.add_argument("--no_packing", action="store_true", help="Disable packing")
    parser.add_argument("--save_total_limit", type=int, default=int(os.environ.get("SAVE_TOTAL_LIMIT", 3)))
    parser.add_argument("--patience_epochs", type=int, default=int(os.environ.get("PATIENCE_EPOCHS", 2)))
    # Testing / subset options
    parser.add_argument("--test_mode", action="store_true", help="Enable quick test mode (small subset, fewer epochs)")
    parser.add_argument("--max_train_samples", type=int, default=int(os.environ.get("MAX_TRAIN_SAMPLES", 0)), help="Limit number of training examples (0 = no limit)")
    parser.add_argument("--max_eval_samples", type=int, default=int(os.environ.get("MAX_EVAL_SAMPLES", 0)), help="Limit number of eval examples (0 = no limit)")
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=int(os.environ.get("LORA_R", 32)))
    parser.add_argument("--lora_alpha", type=int, default=int(os.environ.get("LORA_ALPHA", 64)))
    parser.add_argument("--lora_dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", 0.1)))
    # Class weights for 3-class sentiment (map '0','1','2')
    parser.add_argument(
        "--class_weights",
        type=str,
        default=os.environ.get("CLASS_WEIGHTS", "1.0,5.0,1.0"),
        help="Comma-separated weights for labels 0,1,2 (e.g., '1.0,5.0,1.0')",
    )
    # Weighted sampling (increase neutral frequency in training only)
    parser.add_argument("--weighted_sampler", action="store_true", help="Use WeightedRandomSampler for training")
    parser.add_argument(
        "--target_sampling_ratio",
        type=str,
        default=os.environ.get("TARGET_SAMPLING_RATIO", "0.40,0.20,0.40"),
        help="Target sampling ratio for labels 0,1,2 (e.g., '0.40,0.20,0.40')",
    )
    args = parser.parse_args()
    # Defaults for bf16/packing if not explicitly disabled
    if not args.bf16 and not args.no_bf16:
        args.bf16 = True
    if not args.packing and not args.no_packing:
        args.packing = True
    if args.no_bf16:
        args.bf16 = False
    if args.no_packing:
        args.packing = False
    return args


def assert_files(train_file: str, val_file: str):
    for p in [train_file, val_file]:
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


def attach_lora(base_model, lora_r: int, lora_alpha: int, lora_dropout: float):
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        # Thêm các tham số mới
        inference_mode=False,
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model
class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, *trainer_args, class_weight_map=None, tokenizer=None, **trainer_kwargs):
        super().__init__(*trainer_args, **trainer_kwargs)
        self.class_weight_map = class_weight_map or {}
        self._tok = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [B, T, V]

        # Shift like standard LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute per-token CE loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        vocab_size = shift_logits.size(-1)
        loss_flat = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        loss_per_token = loss_flat.view(shift_labels.size())  # [B, T-1]

        # Mask ignore index
        mask = (shift_labels != -100).float()

        # Build weights: default 1.0, boost the last labeled token per example based on class 0/1/2
        weights = torch.ones_like(loss_per_token)
        with torch.no_grad():
            B, Tm1 = shift_labels.shape
            for b in range(B):
                # indices where label valid
                valid = torch.nonzero(shift_labels[b] != -100, as_tuple=False).squeeze(-1)
                if valid.numel() == 0:
                    continue
                last_idx = valid[-1].item()
                target_id = shift_labels[b, last_idx].item()
                label_char = None
                if self._tok is not None:
                    try:
                        label_char = self._tok.convert_ids_to_tokens([target_id])[0]
                    except Exception:
                        label_char = None
                # Fallback: map token id of ASCII '0','1','2' if available
                weight = 1.0
                if label_char in self.class_weight_map:
                    weight = self.class_weight_map[label_char]
                else:
                    # Try by decoded string of vocab id
                    for k in self.class_weight_map.keys():
                        if isinstance(k, str) and k.isdigit():
                            # If the token ID equals tokenizer for that digit
                            try:
                                kid = self._tok.convert_tokens_to_ids(k)
                                if kid == target_id:
                                    weight = self.class_weight_map[k]
                                    break
                            except Exception:
                                pass
                weights[b, last_idx] = weight

        weighted_loss = (loss_per_token * mask * weights).sum() / (mask.sum() + 1e-8)
        return (weighted_loss, outputs) if return_outputs else weighted_loss



def main():
    args = parse_args()

    data_dir = args.data_dir
    train_file = os.path.join(data_dir, args.train_file)
    val_file = os.path.join(data_dir, args.val_file)
    output_dir = args.output_dir

    assert_files(train_file, val_file)

    tokenizer = get_tokenizer(args.model_id)

    # Load datasets
    train_ds = load_jsonl_as_hf_dataset(train_file)
    val_ds = load_jsonl_as_hf_dataset(val_file)

    # Subset for testing or user-defined limits BEFORE formatting
    if args.test_mode and args.max_train_samples == 0:
        args.max_train_samples = 256
    if args.test_mode and args.max_eval_samples == 0:
        args.max_eval_samples = 256
    if args.max_train_samples and args.max_train_samples > 0 and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples and args.max_eval_samples > 0 and len(val_ds) > args.max_eval_samples:
        val_ds = val_ds.select(range(args.max_eval_samples))

    # Prepare into single text field for SFT
    fmt_cfg = FormattingConfig()
    train_ds = prepare_dataset(train_ds, tokenizer, fmt_cfg)
    val_ds = prepare_dataset(val_ds, tokenizer, fmt_cfg)

    model = get_model(args.model_id)
    model = attach_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # Hyperparameters tối ưu cho format instruction mới
    effective_epochs = args.epochs
    effective_log_steps = args.log_steps
    effective_save_limit = args.save_total_limit
    if args.test_mode:
        effective_epochs = min(args.epochs, 0.1)
        effective_log_steps = 1
        effective_save_limit = 1

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=effective_epochs,
        logging_steps=effective_log_steps,
        save_total_limit=effective_save_limit,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to=args.report_to,
        optim=args.optim,
        packing=args.packing,
        # Thêm các tham số mới để tối ưu
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Khi bật weighted_sampler, cần tắt packing để đảm bảo chỉ số dataset
    # khớp với độ dài weights của sampler, tránh lỗi out-of-bounds.
    if args.weighted_sampler and training_args.packing:
        training_args.packing = False
        tqdm.write("packing disabled because weighted_sampler is enabled")

    def _parse_text_to_prompt_and_label(text: str) -> (str, str):
        # text format: instruction + \n\n + optional input + \n\n + output
        parts = text.split("\n\n")
        if not parts:
            return text, ""
        label = parts[-1].strip()
        prompt_core = " ".join(p.strip() for p in parts[:-1] if p.strip())
        return prompt_core, label

    def _extract_label_with_allowed(gen_text: str, allowed: str) -> str:
        cleaned = gen_text.strip().replace('"', '').replace("'", "").strip()
        for ch in cleaned:
            if ch in allowed:
                return ch
        for ch in allowed:
            if f" {ch}" in cleaned:
                return ch
        return ""

    def _compute_multi_metrics(y_true, y_pred, labels_sorted):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_sorted, average='macro', zero_division=0
        )
        report = classification_report(y_true, y_pred, labels=labels_sorted, digits=4, zero_division=0)
        return acc, prec, rec, f1, report

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
            # infer allowed labels from val set
            labels_seen = set()
            for ex in self.val_dataset:
                text = ex["text"]
                _, label = _parse_text_to_prompt_and_label(text)
                if label in {"0", "1", "2", "3"}:
                    labels_seen.add(label)
            self.allowed = ''.join(sorted(labels_seen)) or '01'

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
            model.eval()
            y_true = []
            y_pred = []

            for ex in tqdm(self.val_dataset, desc=f"validate epoch {int(state.epoch) if state.epoch is not None else '?'}", leave=False):
                text = ex["text"]
                prompt_core, label = _parse_text_to_prompt_and_label(text)
                if label not in set(self.allowed):
                    continue
                # build constrained hint
                if self.allowed == '01':
                    hint = " Trả lời chỉ 0 hoặc 1:"
                elif self.allowed == '012':
                    hint = " Trả lời chỉ 0, 1 hoặc 2:"
                elif self.allowed == '0123':
                    hint = " Trả lời chỉ 0, 1, 2 hoặc 3:"
                else:
                    hint = f" Trả lời chỉ {', '.join(list(self.allowed))}:"
                prompt = f"{prompt_core}{hint}"

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
                pred = _extract_label_with_allowed(gen, self.allowed)
                if pred not in set(self.allowed):
                    # fallback: mark as wrong
                    pred = ""

                y_true.append(label)
                y_pred.append(pred)

            # filter out invalid preds
            pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in set(self.allowed) and p in set(self.allowed)]
            if not pairs:
                tqdm.write("No valid predictions for evaluation.")
                return
            y_true_f, y_pred_f = zip(*pairs)
            labels_sorted = sorted(list(set(self.allowed)))
            acc, prec, rec, f1, report = _compute_multi_metrics(list(y_true_f), list(y_pred_f), labels_sorted)
            tqdm.write(
                f"val epoch={state.epoch:.2f} acc={acc:.4f} f1_macro={f1:.4f} prec_macro={prec:.4f} rec_macro={rec:.4f}"
            )
            tqdm.write("Per-class report:\n" + report)

            # Save best by macro F1
            if f1 > self.best_f1:
                self.best_f1 = f1
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

    # Parse class weights mapping for '0','1','2'
    cw_vals = [v.strip() for v in args.class_weights.split(',')]
    if len(cw_vals) >= 3:
        try:
            cw_map = {"0": float(cw_vals[0]), "1": float(cw_vals[1]), "2": float(cw_vals[2])}
        except Exception:
            cw_map = {"0": 1.0, "1": 1.0, "2": 1.0}
    else:
        cw_map = {"0": 1.0, "1": 1.0, "2": 1.0}

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=format_text_field,
        class_weight_map=cw_map,
        tokenizer=tokenizer,
    )

    # Optional: apply weighted sampler to increase neutral frequency in training
    if args.weighted_sampler:
        # Access raw labels from the unformatted dataset we prepared earlier (before mapping)
        # However after prepare_dataset(), original columns removed. So reload raw for labels.
        raw_train = load_jsonl_as_hf_dataset(train_file)
        labels: List[str] = list(raw_train["output"]) if "output" in raw_train.column_names else []
        # Build sampling weights towards target ratio (0,1,2)
        label_to_index = {"0": 0, "1": 1, "2": 2}
        counts = np.zeros(3, dtype=np.float64)
        for lab in labels:
            if lab in label_to_index:
                counts[label_to_index[lab]] += 1
        counts[counts == 0] = 1.0
        target = np.array([float(x) for x in args.target_sampling_ratio.split(',')], dtype=np.float64)
        if target.size != 3:
            target = np.array([0.40, 0.20, 0.40], dtype=np.float64)
        target = target / target.sum()
        curr = counts / counts.sum()
        scale = target / (curr + 1e-9)
        class_weights_for_sampling = scale / scale.sum()
        sample_weights = np.array([class_weights_for_sampling[label_to_index.get(l, 0)] for l in labels], dtype=np.float64)

        # Create a DataLoader with custom sampler and replace trainer's train_dataloader
        # Khi không packing, kích thước dataset ~ số mẫu; dùng min để an toàn
        actual_dataset_size = len(trainer.train_dataset)
        num_samples = int(min(actual_dataset_size, sample_weights.shape[0]))
        if num_samples <= 0:
            num_samples = actual_dataset_size
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        train_dataloader = DataLoader(
            trainer.train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=trainer.data_collator,
        )
        trainer.get_train_dataloader = lambda: train_dataloader

    trainer.add_callback(TQDMCallback())
    trainer.add_callback(EpochEvalCallback(val_ds, tokenizer, output_dir, patience_epochs=args.patience_epochs))

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()


