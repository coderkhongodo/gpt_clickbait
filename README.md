# GPT-OSS 20B + QLoRA – Vietnamese Sentiment (UIT-VSFC)

End-to-end pipeline to fine-tune `openai/gpt-oss-20b` on UIT-VSFC for 3-class sentiment (0=negative, 1=neutral, 2=positive), with weighted loss, constrained decoding, and full evaluation.

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

### 2) Chuẩn bị dữ liệu (UIT-VSFC)

Đặt dữ liệu gốc ở:

```
data/uit-vsfc/
  train/{sents.txt, sentiments.txt, topics.txt}
  dev/{sents.txt, sentiments.txt, topics.txt}
  test/{sents.txt, sentiments.txt, topics.txt}
```

Sinh instruction 3 lớp (kèm “Đáp án:”):

```bash
python src/processing/prepare_vsfc_sentiment.py
```

Đầu ra:

```
data_processed/jsonl_text_vsfc_sentiment/
  train_instruction.jsonl
  val_instruction.jsonl
  test_instruction.jsonl
```

## 🛠️ Huấn luyện (QLoRA)

Script: `src/train/train_qlora_gpt_oss_20b.py` (argparse đầy đủ).

Ví dụ train chuẩn 3 lớp với class-weights (neutral=5):

```bash
python src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_vsfc_sentiment \
  --output_dir models/gpt-oss-20b-qlora-sent-3cls \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 \
  --lr 5e-4 --epochs 3 --log_steps 10 \
  --optim paged_adamw_8bit --report_to none \
  --warmup_ratio 0.1 --save_total_limit 3 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 \
  --class_weights "1.0,5.0,1.0"
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

### Class weights + sampling (đề xuất cho sentiment – neutral ~4.3%)

Hai kỹ thuật bổ trợ nhau (không trùng lặp):

- Class weights: phạt lỗi của lớp neutral mạnh hơn → gradient cải thiện mỗi lần neutral xuất hiện.
- Oversampling/weighted sampling: tăng tần suất neutral trong batch → mô hình “thấy biên” đủ để học. Với `batch_size=1`, nếu không tăng tần suất, có thể phải chạy hàng chục bước mới gặp neutral → weights khó phát huy.

Khuyến nghị thực dụng (train-only, giữ nguyên dev/test):

- Mục tiêu “tần suất neutral” ~15–20% bước train.
- Class weights khởi điểm: `neg=1.0, neutral=5.0, pos=1.0` (thử grid 3/5/7 cho neutral).
- Sampling weights (xác suất lấy mẫu theo lớp) hướng tới tỉ lệ ~ `neg:neu:pos = 0.4:0.2:0.4`.

Ví dụ dùng `WeightedRandomSampler` (minh hoạ):

```python
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# y: list/array nhãn 0/1/2 theo thứ tự mẫu trong tập train
cls_counts = np.bincount(y, minlength=3)  # [cnt_neg, cnt_neu, cnt_pos]
target_ratio = np.array([0.40, 0.20, 0.40])

curr_ratio = cls_counts / cls_counts.sum()
scale = target_ratio / (curr_ratio + 1e-9)
class_weights_for_sampling = scale / scale.sum()

sample_weights = np.array([class_weights_for_sampling[label] for label in y], dtype=np.float64)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

loader = DataLoader(train_dataset, batch_size=1, sampler=sampler)
```

Ghi chú:
- Khi dùng TRL SFTTrainer, để áp sampler tuỳ biến cần bọc dataset hoặc tự tạo `Trainer`/`DataLoader` ngoài. Nếu muốn, bạn có thể mở PR để tích hợp cờ `--weighted_sampler` vào pipeline.

Tích hợp sẵn trong script train (đã hỗ trợ cờ):

```bash
python src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_vsfc_sentiment \
  --output_dir models/gpt-oss-20b-qlora-sent-3cls-balanced \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 \
  --lr 5e-4 --epochs 3 \
  --class_weights "1.0,5.0,1.0" \
  --weighted_sampler --target_sampling_ratio "0.40,0.20,0.40"
```

## 🔎 Suy luận (constrained decoding tuỳ chọn)

Script: `src/interface/inference_gpt_oss_20b.py`

```bash
python src/interface/inference_gpt_oss_20b.py --constrained --allowed_labels 012
```

Tuỳ chọn:
- `--allowed_labels`: chuỗi nhãn cho phép (vd: `012` cho 3 lớp).
- `--num_samples`: số lượng mẫu hiển thị từ file test nhỏ (nếu dùng `jsonl_text_small/test.jsonl`).

## 📈 Đánh giá trên tập test

Script: `src/eval/evaluate_model.py`

```bash
python src/eval/evaluate_model.py \
  --model_id openai/gpt-oss-20b \
  --adapter_dir models/gpt-oss-20b-qlora-sent-3cls/best \
  --test_file data_processed/jsonl_text_vsfc_sentiment/test_instruction.jsonl \
  --allowed_labels 012 \
  --max_samples 0 \
  --output_csv eval_results_vsfc.csv \
  --summary_json eval_summary_vsfc.json \
  --report_txt classification_report_vsfc.txt \
  --cm_csv confusion_matrix_vsfc.csv
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

### 4) Đánh giá topic

Sử dụng evaluator chung (hỗ trợ `--allowed_labels 0123`):

```bash
python src/eval/evaluate_model.py \
  --model_id openai/gpt-oss-20b \
  --adapter_dir models/gpt-oss-20b-qlora-topic-4cls/best \
  --test_file data_processed/jsonl_text_vsfc_topic/test_instruction.jsonl \
  --allowed_labels 0123 \
  --output_csv eval_results_topic.csv \
  --summary_json eval_summary_topic.json \
  --report_txt classification_report_topic.txt \
  --cm_csv confusion_matrix_topic.csv
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
data_processed/jsonl_text_vsfc_sentiment/{train,val,test}_instruction.jsonl
models/gpt-oss-20b-qlora-*/best/
src/
  analysis/sentiment_eda_vsfc.py
  eval/evaluate_model.py
  interface/inference_gpt_oss_20b.py
  processing/prepare_vsfc_sentiment.py
  train/train_qlora_gpt_oss_20b.py
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