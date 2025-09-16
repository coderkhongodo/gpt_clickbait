# Báo cáo phân loại cảm xúc (Sentiment) – GPT-OSS 20B + QLoRA trên UIT-VSFC

## 1) Mục tiêu
Phân loại cảm xúc tiếng Việt nhị phân trên UIT-VSFC:
- 0: negative (tiêu cực)
- 1: positive (tích cực)

Neutral (1 trong dữ liệu gốc) đã được loại bỏ trong bước chuẩn bị dữ liệu.

## 2) Dữ liệu và chuẩn hóa
- Nguồn: UIT-VSFC (`uit-vsfc/`).
- Từ dữ liệu gốc: `sents.txt` (câu) và `sentiments.txt` (0/1/2 tương ứng negative/neutral/positive).
- Tiền xử lý với `prepare_vsfc_sentiment.py`:
  - Loại bỏ các mẫu neutral (nhãn 1).
  - Ánh xạ: 0→0 (negative), 2→1 (positive).
  - Định dạng instruction JSONL: `instruction` | `input` | `output`.
  - Instruction: "Hãy phân loại cảm xúc câu sau là 0 (tiêu cực) hoặc 1 (tích cực):"

Bộ dữ liệu kết quả (đầy đủ):
- `jsonl_text/train_instruction.jsonl`: 10,968
- `jsonl_text/val_instruction.jsonl`: 1,510
- `jsonl_text/test_instruction.jsonl`: 2,999

Ngoài ra có bộ nhỏ để thử nhanh (`jsonl_text_small/`):
- Train: 1,000 (500 neg + 500 pos), Val: 200, Test prompts: 10

## 3) Mô hình và thiết lập
- Base model: `openai/gpt-oss-20b`.
- Kỹ thuật: QLoRA (PEFT) + 4-bit (bitsandbytes) nếu base không có pre-quant.
- Tokenizer: pad_token = eos_token nếu thiếu.
- Huấn luyện bằng `train_qlora_gpt_oss_20b.py` (TRL SFTTrainer) với các điểm chính:
  - LoRA: r=32, alpha=64, dropout=0.1, target_modules="all-linear".
  - bf16, gradient checkpointing, packing=True.
  - Callback: progress bar; đánh giá cuối mỗi epoch (nếu bật), lưu best theo macro-F1, early stopping (patience=2).

Ví dụ biến môi trường (full set):
```bash
export MODEL_ID="openai/gpt-oss-20b"
export DATA_DIR="jsonl_text"
export OUTPUT_DIR="gpt-oss-20b-qlora-finetune-sent"
export BATCH_SIZE="1"; export EVAL_BATCH_SIZE="1"; export GRAD_ACCUM="8"
export LR="5e-4"; export EPOCHS="5"; export LOG_STEPS="10"
export OPTIM="paged_adamw_8bit"; export REPORT_TO="none"
export PATIENCE_EPOCHS="2"

python3 train_qlora_gpt_oss_20b.py
```

## 4) Suy luận và đánh giá
- Suy luận mẫu nhanh (5 câu): `inference_gpt_oss_20b.py`.
- Đánh giá chi tiết: `eval_100_sentiment.py` (in phân bố 0/1 + accuracy + macro F1/precision/recall).

Ví dụ đánh giá toàn bộ test:
```bash
export MODEL_ID="openai/gpt-oss-20b"
export ADAPTER_DIR="gpt-oss-20b-qlora-finetune-sent/best"  # checkpoint tốt nhất
export DATA_DIR="jsonl_text"
export TEST_FILE="jsonl_text/test_instruction.jsonl"
export NUM="1000000"   # đủ lớn để quét hết
python3 eval_100_sentiment.py
```

## 5) Kết quả
- Test metrics (bạn cung cấp):
  - acc=0.9663
  - f1_macro=0.9663
  - prec_macro=0.9658
  - recall_macro=0.9672
- Phân bố dự đoán (0/1) xem log `Pred counts` trong script đánh giá.

## 6) Thiết kế prompt và hậu xử lý
- Prompt ép mô hình trả về duy nhất 0 hoặc 1 (thêm "Trả lời chỉ 0 hoặc 1:" vào cuối prompt).
- Hậu xử lý: lấy kí tự đầu tiên thuộc {0,1} từ đoạn sinh để loại bỏ ký tự thừa.

## 7) Tái lập thí nghiệm
1) Cài đặt: `pip install -r requirements.txt`.
2) Sinh dữ liệu (nếu cần): `python3 prepare_vsfc_sentiment.py`.
3) Huấn luyện: chạy khối lệnh ở Mục 3.
4) Đánh giá: chạy khối lệnh ở Mục 4.

## 8) Hướng mở rộng
- Thêm few-shot ví dụ vào instruction.
- Điều chỉnh `packing`, `GRAD_ACCUM`, `LR`, `EPOCHS` theo tài nguyên.
- Báo cáo thêm confusion matrix/ROC nếu cần.
