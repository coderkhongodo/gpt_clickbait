# GPT-OSS 20B + QLoRA – Vietnamese Clickbait Detection

End-to-end pipeline tinh chỉnh `openai/gpt-oss-20b` cho phân loại clickbait tiếng Việt:
- Phân loại Clickbait (2 lớp: 0=không phải clickbait, 1=clickbait)
- Sử dụng dataset clickbait tiếng Việt với 3,414 mẫu
- Kết hợp tiêu đề và nội dung bài báo để phân loại

Dataset bao gồm:
- 2,349 mẫu không phải clickbait (68.8%)
- 1,065 mẫu clickbait (31.2%)
- Nguồn: Các trang báo tiếng Việt (VnExpress, Tuổi Trẻ, SaoStar, v.v.)

## 🚀 Quick Start

### 1) Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu phần cứng/phần mềm:
- Python 3.10+
- GPU: A6000 Ada 48GB (đã kiểm thử tốt). Các GPU VRAM ≥ 24GB cũng có thể chạy khi bật QLoRA + grad accumulation.
- CUDA: 12.8 (PyTorch 2.2+ hỗ trợ CUDA 12.x). Cài đặt theo hướng dẫn PyTorch chính thức tương ứng hệ điều hành.

Gợi ý cài PyTorch phù hợp CUDA 12.x (Windows/Linux, pip):

```bash
# CUDA 12.x build (khuyên dùng):
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Nếu dùng CPU-only (không khuyến nghị để train):
# pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

Lưu ý:
- CUDA hệ thống 12.8 tương thích với build cu121 của PyTorch.
- bitsandbytes>=0.43.0 hỗ trợ CUDA 12.x; trên Windows cần bản Python 3.10+.

### 2) Chuẩn bị dữ liệu (Clickbait Dataset)

Đặt dữ liệu gốc ở:

```
data/data_raw/
  clickbait_dataset_vietnamese.csv
  clickbait_dataset_vietnamese.jsonl
```

Sinh instruction 3 lớp (kèm “Đáp án:”):

Chọn một trong hai (hoặc cả hai):

- Sentiment 2 lớp (khuyến nghị):
```bash
python src/processing/prepare_vsfc_sentiment_2cls.py
```
Đầu ra: `data_processed/jsonl_text_vsfc_sentiment_2cls/{train,val,test}_instruction.jsonl`

- Topic 4 lớp:
```bash
python src/processing/prepare_vsfc_topic.py
```
Đầu ra: `data_processed/jsonl_text_vsfc_topic/{train,val,test}_instruction.jsonl`

## 🛠️ Huấn luyện (QLoRA)

Script: `src/train/train_qlora_gpt_oss_20b.py` (argparse đầy đủ).

### Sentiment 2 lớp (0=neg, 1=pos)
```bash
python3 src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_vsfc_sentiment_2cls \
  --output_dir models/gpt-oss-20b-qlora-sent-2cls \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 \
  --lr 5e-4 --epochs 10  --log_steps 10 \
  --optim paged_adamw_8bit --report_to none \
  --warmup_ratio 0.1 --save_total_limit 3 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.1
```

Chế độ thử nhanh (subset + epochs nhỏ):

```bash
python src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_vsfc_sentiment \
  --output_dir models/gpt-oss-20b-qlora-sent-3cls-test \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 4 \
  --lr 5e-4 --epochs 1 --test_mode
```

Notes cấu hình trên A6000 Ada 48GB + CUDA 12.8:
- Mặc định script tự phát hiện pre-quant (MXFP4). Nếu base model không có pre-quant, sẽ fallback 4-bit (bitsandbytes) để vừa VRAM.
- Khuyến nghị: `--batch_size 1 --grad_accum 16..32`, `--bf16` bật sẵn; có thể tăng `--epochs` tuỳ thời gian.
- Weighted loss áp tại token nhãn cuối cùng (mapping nhãn '0','1','2').

Ghi chú: Với 2 lớp đã loại bỏ neutral, không cần sampler/class-weight đặc biệt.

## 🔎 Suy luận (constrained decoding)

Script: `src/interface/inference_gpt_oss_20b.py`

```bash
# Sentiment 2 lớp
python src/interface/inference_gpt_oss_20b.py --constrained --allowed_labels 01

# Topic 4 lớp
python src/interface/inference_gpt_oss_20b.py --constrained --allowed_labels 0123
```

Tuỳ chọn:
- `--allowed_labels`: chuỗi nhãn cho phép (vd: `012` cho 3 lớp).
- `--num_samples`: số lượng mẫu hiển thị từ file test nhỏ (nếu dùng `jsonl_text_small/test.jsonl`).

## 📈 Đánh giá trên tập test (lưu vào results/)

Script: `src/eval/evaluate_model.py`

```bash
# Sentiment 2 lớp
mkdir -p results && python src/eval/evaluate_model.py \
  --model_id openai/gpt-oss-20b \
  --adapter_dir models/gpt-oss-20b-qlora-sent-2cls/best \
  --test_file data_processed/jsonl_text_vsfc_sentiment_2cls/test_instruction.jsonl \
  --allowed_labels 01 \
  --output_csv results/eval_results_sent2cls_test.csv \
  --summary_json results/eval_summary_sent2cls_test.json \
  --report_txt results/classification_report_sent2cls_test.txt \
  --cm_csv results/confusion_matrix_sent2cls_test.csv
```

In ra và lưu:
- Accuracy, Precision/Recall/F1 (macro)
- Classification report (txt)
- Confusion matrix (CSV)
- Kết quả chi tiết từng mẫu (CSV) và summary (JSON)

## 🧩 Bài toán Topic (UIT-VSFC – 4 lớp)

### 1) Chuẩn bị dữ liệu topic

```bash
python src/processing/prepare_vsfc_topic.py
```

Đầu ra:

```
data_processed/jsonl_text_vsfc_topic/
  train_instruction.jsonl
  val_instruction.jsonl
  test_instruction.jsonl
```

### 2) Huấn luyện topic (QLoRA)

```bash
python src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_vsfc_topic \
  --output_dir models/gpt-oss-20b-qlora-topic-4cls \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 \
  --lr 5e-4 --epochs 3 --log_steps 10 \
  --optim paged_adamw_8bit --report_to none \
  --warmup_ratio 0.1 --save_total_limit 3 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.1
```

Chế độ thử nhanh:

```bash
python src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_vsfc_topic \
  --output_dir models/gpt-oss-20b-qlora-topic-4cls-test \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 4 \
  --lr 5e-4 --epochs 1 --test_mode
```

### 3) Suy luận topic (ràng buộc 0/1/2/3)

```bash
python src/interface/inference_gpt_oss_20b.py --constrained --allowed_labels 0123
```

### 4) Đánh giá topic (lưu vào results/)

Sử dụng evaluator chung (hỗ trợ `--allowed_labels 0123`):

```bash
mkdir -p results && python src/eval/evaluate_model.py \
  --model_id openai/gpt-oss-20b \
  --adapter_dir models/gpt-oss-20b-qlora-topic-4cls/best \
  --test_file data_processed/jsonl_text_vsfc_topic/test_instruction.jsonl \
  --allowed_labels 0123 \
  --output_csv results/eval_results_topic_test.csv \
  --summary_json results/eval_summary_topic_test.json \
  --report_txt results/classification_report_topic_test.txt \
  --cm_csv results/confusion_matrix_topic_test.csv
```

### 5) Gợi ý class weights cho topic (mất cân bằng)

Theo phân bố ví dụ (0:11607, 1:3040, 2:712, 3:816), có thể bắt đầu với:

- Khuyến nghị (sqrt-inverse, ổn định): `--class_weights "1.0,2.0,4.1,3.8"`
- Mạnh hơn (inverse freq, có giới hạn): `--class_weights "1.0,3.5,6.0,5.5"`

Tip: bắt đầu với bộ sqrt-inverse; nếu lớp hiếm (2,3) còn yếu, tăng dần trọng số 2→4.5 và 3→4.2. Luôn bật constrained decoding khi suy luận: `--constrained --allowed_labels 0123`.

## 📊 Phân tích dữ liệu

Script EDA: `src/analysis/sentiment_eda_vsfc.py`

```bash
python src/analysis/sentiment_eda_vsfc.py --vsfc_dir data/uit-vsfc --save_plots --save_preview
```

Sinh biểu đồ phân bố nhãn, độ dài; preview JSONL 3 lớp.

## 🧭 Cấu trúc dự án (rút gọn)

```
data/uit-vsfc/...
data_processed/
  jsonl_text_vsfc_sentiment_2cls/{train,val,test}_instruction.jsonl
  jsonl_text_vsfc_topic/{train,val,test}_instruction.jsonl
models/gpt-oss-20b-qlora-*/best/
src/
  analysis/sentiment_eda_vsfc.py
  eval/evaluate_model.py
  interface/inference_gpt_oss_20b.py
  processing/prepare_vsfc_sentiment_2cls.py
  processing/prepare_vsfc_topic.py
  train/train_qlora_gpt_oss_20b.py
docs/
  sentiment_classification_report.md
  topic_classification_report.md
```

## 📋 Requirements

```
torch>=2.1.0
transformers>=4.42.0
peft>=0.11.1
trl>=0.9.4
datasets>=2.20.0
accelerate>=0.31.0
bitsandbytes>=0.43.0
scikit-learn>=1.3.0
pandas>=2.0.0
```