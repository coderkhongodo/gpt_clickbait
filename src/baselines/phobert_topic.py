import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd


@dataclass
class Config:
    model_id: str = "vinai/phobert-base"
    vsfc_dir: str = os.path.join("data", "uit-vsfc")
    output_dir: str = os.path.join("models", "phobert-topic-4cls")
    results_dir: str = "results"
    epochs: int = 15
    lr: float = 2e-5
    batch_size: int = 16
    max_length: int = 160
    seed: int = 42
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_steps: int = 200
    logging_steps: int = 50
    save_total_limit: int = 2


def read_split_text_and_labels(split_dir: str) -> (List[str], List[int]):
    sents_path = os.path.join(split_dir, "sents.txt")
    labels_path = os.path.join(split_dir, "topics.txt")
    with open(sents_path, "r", encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f]
    if len(texts) != len(labels):
        raise ValueError(f"Mismatched lines: {sents_path} vs {labels_path}")
    return texts, labels


def load_vsfc_as_hf_dataset(vsfc_dir: str) -> DatasetDict:
    data = {}
    for split in ["train", "dev", "test"]:
        split_dir = os.path.join(vsfc_dir, split)
        texts, labels = read_split_text_and_labels(split_dir)
        data[split] = Dataset.from_dict({"text": texts, "label": labels})
    return DatasetDict(data)


def tokenize_function(examples: Dict, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
    }


def save_test_reports(preds: np.ndarray, labels: np.ndarray, results_dir: str, prefix: str = "phobert_topic"):
    os.makedirs(results_dir, exist_ok=True)

    # Summary metrics
    acc = accuracy_score(labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    summary = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
    }
    with open(os.path.join(results_dir, f"{prefix}_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Classification report
    report = classification_report(labels, preds, digits=4, zero_division=0)
    with open(os.path.join(results_dir, f"{prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1", "true_2", "true_3"], columns=["pred_0", "pred_1", "pred_2", "pred_3"]) 
    cm_df.to_csv(os.path.join(results_dir, f"{prefix}_confusion_matrix.csv"), index=True, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser("PhoBERT baseline for UIT-VSFC Topic (4-class)")
    parser.add_argument("--model_id", type=str, default=os.environ.get("PHOBERT_MODEL", "vinai/phobert-base"))
    parser.add_argument("--vsfc_dir", type=str, default=os.environ.get("VSFC_DIR", os.path.join("data", "uit-vsfc")))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", os.path.join("models", "phobert-topic-4cls")))
    parser.add_argument("--results_dir", type=str, default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 5)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", 2e-5)))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 16)))
    parser.add_argument("--max_length", type=int, default=int(os.environ.get("MAX_LENGTH", 160)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    args = parser.parse_args()

    cfg = Config(
        model_id=args.model_id,
        vsfc_dir=args.vsfc_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    dsd = load_vsfc_as_hf_dataset(cfg.vsfc_dir)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_id, num_labels=4)

    # Tokenize
    tokenized = dsd.map(lambda ex: tokenize_function(ex, tokenizer, cfg.max_length), batched=True)
    tokenized = tokenized.remove_columns(["text"]).with_format("torch")

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        logging_steps=cfg.logging_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        report_to=["none"],
        fp16=torch.cuda.is_available(),
        save_total_limit=cfg.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    class EarlyStopEvalCallback(TrainerCallback):
        def __init__(self, val_dataset, patience_epochs: int = 3):
            self.val_dataset = val_dataset
            self.best_f1 = -1.0
            self.no_improve = 0
            self.patience = patience_epochs

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            trainer.model.eval()
            with torch.no_grad():
                out = trainer.predict(self.val_dataset)
            logits = out.predictions
            labels = out.label_ids
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            print(f"val epoch={state.epoch:.2f} acc={acc:.4f} f1_macro={f1_macro:.4f} prec_macro={prec_macro:.4f} rec_macro={rec_macro:.4f}")
            # Save best
            if f1_macro > self.best_f1:
                self.best_f1 = f1_macro
                self.no_improve = 0
                save_dir = os.path.join(args.output_dir, "best")
                os.makedirs(save_dir, exist_ok=True)
                trainer.save_model(save_dir)
                if trainer.tokenizer is not None:
                    trainer.tokenizer.save_pretrained(save_dir)
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    control.should_training_stop = True

    trainer.add_callback(EarlyStopEvalCallback(tokenized["dev"], patience_epochs=3))

    trainer.train()

    # Load best checkpoint if exists
    best_dir = os.path.join(cfg.output_dir, "best")
    if os.path.isdir(best_dir):
        model = AutoModelForSequenceClassification.from_pretrained(best_dir)
        trainer.model = model
        try:
            trainer._move_model_to_device(trainer.model, trainer.args.device)
        except Exception:
            trainer.model.to(trainer.args.device)

    # Evaluate on test using best
    with torch.no_grad():
        preds_output = trainer.predict(tokenized["test"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    # Save reports
    save_test_reports(preds, labels, cfg.results_dir, prefix="phobert_topic")

    # Save model
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()


