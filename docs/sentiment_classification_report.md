## Báo cáo chi tiết: Phân loại Sentiment UIT-VSFC (2 lớp)

### 1. Mục tiêu
Huấn luyện và đánh giá GPT-OSS 20B (QLoRA) cho bài toán phân loại cảm xúc tiếng Việt trên UIT-VSFC với cấu hình 2 lớp: 0=negative, 1=positive (đã loại bỏ neutral).

### 2. Dữ liệu
- Nguồn: `data/uit-vsfc/{train,dev,test}/(sents.txt, sentiments.txt)`
- Script xử lý: `src/processing/prepare_vsfc_sentiment_2cls.py`
- Đầu ra: `data_processed/jsonl_text_vsfc_sentiment_2cls/{train,val,test}_instruction.jsonl`

Sinh dữ liệu:
```bash
# 2 lớp (bỏ neutral, map 2->1)
python3 src/processing/prepare_vsfc_sentiment_2cls.py
```

### 3. Định dạng mẫu (instruction)
```json
{
  "instruction": "Bạn là bộ phân loại cảm xúc... (mô tả nhãn)",
  "input": "<câu văn>\nĐáp án:",
  "output": "<0|1>"
}
```

### 4. Huấn luyện
- Base model: `openai/gpt-oss-20b`
- QLoRA: LoRA r=32, alpha=64, dropout=0.1; bfloat16; gradient checkpointing; optimizer `paged_adamw_8bit`
- Packing mặc định bật (nhanh). Nếu bật weighted sampler (3 lớp), code tự tắt packing để tránh sai lệch chỉ số.

#### 4.1. Huấn luyện 2 lớp (không dùng sampler)
Không cần sampler; class weights để mặc định.
```bash
cd /root/gpt_oss_sematic_sentiment && python3 src/train/train_qlora_gpt_oss_20b.py \\
  --model_id openai/gpt-oss-20b \\
  --data_dir data_processed/jsonl_text_vsfc_sentiment_2cls \\
  --output_dir models/gpt-oss-20b-qlora-sent-2cls \\
  --train_file train_instruction.jsonl \\
  --val_file val_instruction.jsonl \\
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 \\
  --lr 5e-4 --epochs 10  --log_steps 10 \\
  --optim paged_adamw_8bit --report_to none \\
  --warmup_ratio 0.1 --save_total_limit 3 \\
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 \\
  --class_weights "1.0,1.0,1.0"
```

### 5. Suy luận (constrained decoding)
- 3 lớp: `--allowed_labels 012`
- 2 lớp: `--allowed_labels 01`

```bash
# ví dụ 3 lớp
python3 src/interface/inference_gpt_oss_20b.py --constrained --allowed_labels 012
```

### 6. Đánh giá (đầy đủ chỉ số) – ghi ra `results/`
Script: `src/eval/evaluate_model.py` — in Accuracy, Precision/Recall/F1 (macro/weighted/micro), classification report, confusion matrix.

#### 6.1. 2 lớp
```bash
mkdir -p results && python3 src/eval/evaluate_model.py \\
  --model_id openai/gpt-oss-20b \\
  --adapter_dir models/gpt-oss-20b-qlora-sent-2cls/best \\
  --test_file data_processed/jsonl_text_vsfc_sentiment_2cls/test_instruction.jsonl \\
  --allowed_labels 01 \\
  --output_csv results/eval_results_sent2cls_test.csv \\
  --summary_json results/eval_summary_sent2cls_test.json \\
  --report_txt results/classification_report_sent2cls_test.txt \\
  --cm_csv results/confusion_matrix_sent2cls_test.csv
```

### 7. Kết quả test (2 lớp) — lấy từ `results/`

Tóm tắt `results/eval_summary_sent2cls_test.json`:

- Tổng mẫu: 2,999 — Accuracy: 96.80%
- F1: Macro 96.79%, Weighted 96.80%, Micro 96.80%

Bảng `results/classification_report_sent2cls_test.txt`:

| Lớp | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative (0) | 95.56 | 97.73 | 96.63 | 1409 |
| Positive (1) | 97.95 | 95.97 | 96.95 | 1590 |
| Accuracy |  |  | 96.80 | 2999 |
| Macro avg | 96.75 | 96.85 | 96.79 | 2999 |
| Weighted avg | 96.82 | 96.80 | 96.80 | 2999 |

### 8. Lưu ý hiệu năng
- Bật packing giúp tốc độ nhanh. Nếu dùng weighted sampler (3 lớp), code tự chuyển sang không packing để an toàn chỉ số → tốc độ chậm hơn; có thể giảm epochs/giới hạn mẫu để thử nhanh.

### 9. Tái lập thí nghiệm
1) Chuẩn bị dữ liệu theo 3 lớp hoặc 2 lớp.
2) Huấn luyện theo lệnh ở mục 4.
3) Đánh giá theo lệnh ở mục 6 để sinh các file trong `results/`.

### 10. Hướng cải thiện
- Với 3 lớp: tăng class weights/sampler cho neutral, thử focal loss, tăng epochs; điều chỉnh lr nhỏ hơn ở giai đoạn finetune.
- Với 2 lớp: có thể tăng `--epochs` để đạt F1 cao hơn và thử regularization nhẹ (dropout/weight decay mặc định của optimizer).


