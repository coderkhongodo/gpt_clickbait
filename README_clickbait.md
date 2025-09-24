# GPT-OSS 20B + QLoRA – Vietnamese Clickbait Detection

End-to-end pipeline tinh chỉnh `openai/gpt-oss-20b` cho phân loại clickbait tiếng Việt:
- Phân loại Clickbait (2 lớp: 0=không phải clickbait, 1=clickbait)
- Sử dụng dataset clickbait tiếng Việt với 3,414 mẫu
- Phân loại dựa trên tiêu đề bài báo (không sử dụng nội dung)

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

Xử lý và chia dataset thành train/val/test:

```bash
python src/processing/prepare_clickbait_dataset.py
```

Đầu ra: `data_processed/jsonl_text_clickbait/{train,val,test}_instruction.jsonl`

Dataset được chia theo tỷ lệ:
- Train: 70% (2,389 mẫu)
- Validation: 15% (512 mẫu)  
- Test: 15% (513 mẫu)

## 🛠️ Huấn luyện (QLoRA)

Script: `src/train/train_qlora_gpt_oss_20b.py` (argparse đầy đủ).

### Clickbait Binary Classification (0=non-clickbait, 1=clickbait)
```bash
python src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_clickbait \
  --output_dir models/gpt-oss-20b-qlora-clickbait \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 \
  --lr 5e-4 --epochs 5 --log_steps 10 \
  --optim paged_adamw_8bit --report_to none \
  --warmup_ratio 0.1 --save_total_limit 3 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 \
  --class_weights "1.0,1.0"
```

Chế độ thử nhanh (subset + epochs nhỏ):

```bash
python src/train/train_qlora_gpt_oss_20b.py \
  --model_id openai/gpt-oss-20b \
  --data_dir data_processed/jsonl_text_clickbait \
  --output_dir models/gpt-oss-20b-qlora-clickbait-test \
  --train_file train_instruction.jsonl \
  --val_file val_instruction.jsonl \
  --batch_size 1 --eval_batch_size 1 --grad_accum 4 \
  --lr 5e-4 --epochs 1 --test_mode
```

## 🔍 Suy luận (Inference)

Script: `src/interface/inference_gpt_oss_20b.py`

```bash
python src/interface/inference_gpt_oss_20b.py --constrained --allowed_labels 01
```

Tuỳ chọn:
- `--allowed_labels`: chuỗi nhãn cho phép (vd: `01` cho 2 lớp).
- `--num_samples`: số lượng mẫu hiển thị từ file test.

## 📈 Đánh giá trên tập test (lưu vào results/)

Script: `src/eval/evaluate_model.py`

```bash
# Clickbait Binary Classification
mkdir -p results && python src/eval/evaluate_model.py \
  --model_id openai/gpt-oss-20b \
  --adapter_dir models/gpt-oss-20b-qlora-clickbait/best \
  --test_file data_processed/jsonl_text_clickbait/test_instruction.jsonl \
  --allowed_labels 01 \
  --output_csv results/eval_results_clickbait_test.csv \
  --summary_json results/eval_summary_clickbait_test.json \
  --report_txt results/classification_report_clickbait_test.txt \
  --cm_csv results/confusion_matrix_clickbait_test.csv
```

## 📊 Cấu trúc thư mục

```
data/data_raw/clickbait_dataset_vietnamese.csv
data_processed/
  jsonl_text_clickbait/{train,val,test}_instruction.jsonl
models/gpt-oss-20b-qlora-clickbait/best/
src/
  eval/evaluate_model.py
  interface/inference_gpt_oss_20b.py
  processing/prepare_clickbait_dataset.py
  train/train_qlora_gpt_oss_20b.py
```

## 📋 Requirements

```
torch>=2.2.0
transformers>=4.42.0
accelerate>=0.31.0
bitsandbytes>=0.43.0
peft>=0.11.1
trl>=0.9.4
datasets>=2.20.0
scikit-learn>=1.3.0
pandas>=2.0.0
seaborn>=0.13.0
matplotlib>=3.7.0
tqdm>=4.66.0
```

## 🎯 Định dạng dữ liệu

### Input format (instruction):
```json
{
  "instruction": "Bạn là bộ phân loại clickbait.\nNhiệm vụ: Xác định xem tiêu đề và nội dung bài báo có phải là clickbait hay không.\nQuy ước:\n- 0 = không phải clickbait: tiêu đề trung thực, mô tả chính xác nội dung bài viết\n- 1 = clickbait: tiêu đề câu view, phóng đại, gây tò mò nhưng không phản ánh đúng nội dung\n\nChỉ trả đúng MỘT ký tự trong {0,1}. Không thêm chữ nào khác.\n\nBài viết cần phân loại:",
  "input": "Tiêu đề: Yêu nhau lắm\nNội dung: - Ông có để ý giờ nhiều cặp đôi cứ chia tay là lên mạng ì xèo không?\nĐáp án:",
  "output": "1"
}
```

### Dataset schema:
- **id**: Unique identifier
- **title**: Tiêu đề bài báo
- **lead_paragraph**: Đoạn mở đầu/tóm tắt nội dung
- **label**: clickbait hoặc non-clickbait
- **category**: Danh mục bài báo
- **source**: Nguồn báo (VnExpress, Tuổi Trẻ, etc.)

## 🔧 Tùy chỉnh

### Class weights cho imbalanced dataset:
```bash
--class_weights "2.0,1.0"  # Tăng trọng số cho class 0 (non-clickbait)
```

### Weighted sampling:
```bash
--weighted_sampler --target_sampling_ratio "0.50,0.50"
```

### Early stopping:
```bash
--patience_epochs 3
```
