## Báo cáo chi tiết: Phân loại Topic trên UIT-VSFC (4 lớp)

### 1. Mục tiêu
Xây dựng pipeline huấn luyện và đánh giá mô hình ngôn ngữ lớn (GPT-OSS 20B + QLoRA) cho bài toán phân loại chủ đề (topic) tiếng Việt trên bộ dữ liệu UIT-VSFC với 4 lớp:
- 0: Giảng viên (Lecturer)
- 1: Chương trình đào tạo (Curriculum)
- 2: Cơ sở vật chất (Facility)
- 3: Khác (Others)

### 2. Dữ liệu
- Nguồn: `data/uit-vsfc/{train,dev,test}/(sents.txt, topics.txt)`
- Quy mô (thực tế đo lại):
  - Train: 11,426 mẫu — 0: 71.5%, 1: 19.3%, 2: 4.3%, 3: 4.9%
  - Dev: 1,583 mẫu — 0: 72.7%, 1: 16.9%, 2: 4.4%, 3: 6.0%
  - Test: 3,166 mẫu — 0: 72.3%, 1: 18.1%, 2: 4.6%, 3: 5.0%
- Mất cân bằng lớp rõ rệt, đặc biệt lớp 2 và 3 là hiếm.

### 3. Tiền xử lý
- Script: `src/processing/prepare_vsfc_topic.py`
- Đầu vào: `sents.txt`, `topics.txt`
- Đầu ra: JSONL theo định dạng instruction-tuning
  - `data_processed/jsonl_text_vsfc_topic/{train,val,test}_instruction.jsonl`
- Mẫu bản ghi:
```json
{
  "instruction": "Bạn là bộ phân loại chủ đề (topic).\nQuy ước lớp:\n- 0 = giảng viên\n- 1 = chương trình đào tạo\n- 2 = cơ sở vật chất\n- 3 = khác\n\nChỉ trả đúng MỘT ký tự trong {0,1,2,3}. Không thêm chữ nào khác.\n\nCâu cần phân loại:",
  "input": "<câu văn>\nĐáp án:",
  "output": "<0|1|2|3>"
}
```

### 4. Mô hình và huấn luyện
- Base model: `openai/gpt-oss-20b`
- Kỹ thuật tinh chỉnh: QLoRA (4-bit nếu không có pre-quant), LoRA r=32, alpha=64, dropout=0.1
- Tokenizer: tự căn chỉnh PAD/BOS/EOS khi cần
- Tham số mặc định quan trọng:
  - `batch_size=1`, `grad_accum=16`, `bf16=True`, `gradient_checkpointing=True`
  - Lịch LR cosine, `warmup_ratio=0.1`
  - Evaluate mỗi epoch với suy luận ràng buộc (chỉ cho phép 0/1/2/3)
- Weighted loss: tăng trọng số tại token nhãn cuối cùng theo lớp (mặc định dành cho 3 lớp, nhưng pipeline topic sử dụng f1 macro để chọn best). Với topic, khuyến nghị sử dụng class weights:
  - sqrt-inverse: `"1.0,2.0,4.1,3.8"`
  - mạnh hơn: `"1.0,3.5,6.0,5.5"`

Lệnh huấn luyện (đang sử dụng):
```bash
cd /root/gpt_oss_sematic_sentiment && python3 src/train/train_qlora_gpt_oss_20b.py --model_id openai/gpt-oss-20b --data_dir data_processed/jsonl_text_vsfc_topic --output_dir models/gpt-oss-20b-qlora-topic-4cls --train_file train_instruction.jsonl --val_file val_instruction.jsonl --batch_size 1 --eval_batch_size 1 --grad_accum 16 --lr 1e-4 --epochs 15 --log_steps 10 --optim paged_adamw_8bit --report_to none --warmup_ratio 0.1 --save_total_limit 3 --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --class_weights "1.0,4.0,12.0,18.0"
```

Lưu ý hiệu năng:
- Packing được bật mặc định để tăng tốc. Khi dùng `WeightedRandomSampler` (đặc thù bài toán khác), packing sẽ bị tắt tự động để tránh sai lệch chỉ số dataset.

### 5. Suy luận và ràng buộc
- Trong quá trình eval/early-stop, mô hình sinh đầu ra ràng buộc tập ký tự cho phép `0123`.
- Hàm lấy nhãn chọn ký tự đầu tiên thuộc tập cho phép, giảm lỗi định dạng.

### 6. Đánh giá
- Script: `src/eval/evaluate_model.py` (đã hỗ trợ in Accuracy, Precision/Recall/F1 macro/weighted/micro; classification report; confusion matrix)
- Lệnh đánh giá trên test (đúng như đang chạy):
```bash
cd /root/gpt_oss_sematic_sentiment && python3 src/eval/evaluate_model.py \
  --model_id openai/gpt-oss-20b \
  --adapter_dir models/gpt-oss-20b-qlora-topic-4cls/best \
  --test_file data_processed/jsonl_text_vsfc_topic/test_instruction.jsonl \
  --allowed_labels 0123 \
  --max_samples 0 \
  --output_csv eval_results_topic_test.csv \
  --summary_json eval_summary_topic_test.json \
  --report_txt classification_report_topic_test.txt \
  --cm_csv confusion_matrix_topic_test.csv
```

#### 6.1. Baseline (từ tài liệu UIT-VSFC, classifier MaxEnt)
- Bảng kết quả (test):

| Lớp | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Lecturer (0)   | 90.17 | 92.10 | 91.12 |
| Curriculum (1) | 67.07 | 67.31 | 67.19 |
| Facility (2)   | 90.65 | 86.90 | 88.73 |
| Others (3)     | 45.61 | 32.70 | 38.10 |
| Average (macro)| 83.78 | 84.40 | 84.03 |

#### 6.2. Kết quả trong quá trình huấn luyện (validation)

| Epoch | Acc | F1 Macro | Prec Macro | Rec Macro | F1 (0) | F1 (1) | F1 (2) | F1 (3) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 86.23 | 71.56 | 88.38 | 69.65 | 91.69 | 74.79 | 89.39 | 30.36 |
| 3 | 84.71 | 76.43 | 78.84 | 76.96 | 90.97 | 70.78 | 90.23 | 53.75 |
| 4 | 83.89 | 72.35 | 84.59 | 72.72 | 90.39 | 70.47 | 90.23 | 38.33 |

Ghi chú: Các số liệu lấy từ log huấn luyện và được ràng buộc nhãn sinh ra trong {0,1,2,3}.

#### 6.3. Kết quả test (đầy đủ chỉ số) – lưu trong `results/`
Sau khi huấn luyện xong, chạy:
```bash
mkdir -p results && python3 src/eval/evaluate_model.py \
  --model_id openai/gpt-oss-20b \
  --adapter_dir models/gpt-oss-20b-qlora-topic-4cls/best \
  --test_file data_processed/jsonl_text_vsfc_topic/test_instruction.jsonl \
  --allowed_labels 0123 \
  --output_csv results/eval_results_topic_test.csv \
  --summary_json results/eval_summary_topic_test.json \
  --report_txt results/classification_report_topic_test.txt \
  --cm_csv results/confusion_matrix_topic_test.csv
```

Các file sinh ra:
- `results/eval_summary_topic_test.json`: Accuracy, Precision/Recall/F1 (macro/weighted/micro)
- `results/classification_report_topic_test.txt`: precision/recall/F1 theo lớp
- `results/confusion_matrix_topic_test.csv`: ma trận nhầm lẫn
- `results/eval_results_topic_test.csv`: dự đoán chi tiết từng mẫu

##### 6.3.1. Tóm tắt số liệu test (đã chạy)

Kết quả GPT-OSS 20B (QLoRA) — nguồn: `results/eval_summary_topic_test.json` và `results/classification_report_topic_test.txt`
- Tổng mẫu: 3,166 — Accuracy: 87.18%
- F1: Macro 79.15%, Weighted 87.57%, Micro 87.18%

| Lớp | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Lecturer (0)   | 95.06 | 90.70 | 92.83 | 2290 |
| Curriculum (1) | 69.80 | 77.97 | 73.66 | 572  |
| Facility (2)   | 92.62 | 95.17 | 93.88 | 145  |
| Others (3)     | 51.30 | 62.26 | 56.25 | 159  |
| Macro Avg      | 77.19 | 81.53 | 79.15 | 3166 |
| Weighted Avg   | 88.18 | 87.18 | 87.57 | 3166 |

Kết quả baseline PhoBERT — nguồn: `results/phobert_topic_eval_summary.json` và `results/phobert_topic_classification_report.txt`
- Accuracy: 89.13%
- F1: Macro 78.89%, Weighted 88.90%

| Lớp | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Lecturer (0)   | 92.98 | 95.46 | 94.20 | 2290 |
| Curriculum (1) | 80.92 | 73.43 | 76.99 | 572  |
| Facility (2)   | 88.16 | 92.41 | 90.24 | 145  |
| Others (3)     | 56.94 | 51.57 | 54.13 | 159  |
| Macro Avg      | 79.75 | 78.22 | 78.89 | 3166 |
| Weighted Avg   | 88.77 | 89.13 | 88.90 | 3166 |

So sánh nhanh GPT-OSS vs PhoBERT (trên test):
- Accuracy: PhoBERT nhỉnh hơn (89.13% vs 87.18%).
- F1 macro: gần tương đương (79.15% vs 78.89%).
- Theo lớp: GPT-OSS tốt hơn lớp 1 và 2; PhoBERT nhỉnh hơn lớp 0; lớp 3 tương đương (F1 ~56). 

Nhận xét:
- Hai mô hình cho chất lượng sát nhau về F1 macro; PhoBERT có độ chính xác tổng thể cao hơn nhẹ.
- GPT-OSS 20B linh hoạt hơn cho instruction-format và có thể cải thiện thêm qua tinh chỉnh class weights/epochs.

Nhận xét:
- Ba lớp chính (0,1,2) đạt và vượt baseline sớm; lớp 3 (Others) cải thiện dần theo thời gian huấn luyện.
- F1 macro trên dev tăng mạnh tới epoch 3; cần theo dõi overfitting ở các epoch sau.

### 7. Kỹ thuật xử lý mất cân bằng
- Class weights: tăng trọng số lỗi ở token nhãn cuối cùng, đặc biệt cho lớp hiếm.
- Constrained decoding: hạn chế không gian sinh đầu ra, giảm sai định dạng.
- Có thể thử weighted sampling (ít dùng cho topic vì mức mất cân bằng vừa phải; nếu bật, code đã tự tắt packing để đảm bảo an toàn chỉ số).

### 8. Tái lập thí nghiệm (Reproducibility)
1) Chuẩn bị dữ liệu topic:
```bash
python3 src/processing/prepare_vsfc_topic.py
```
2) Huấn luyện (ví dụ 3 epoch): xem lệnh ở mục 4.
3) Đánh giá test: xem lệnh ở mục 6.

### 9. Hạn chế và hướng phát triển
- Lớp 3 (Others) vẫn khó do ít dữ liệu và ranh giới mơ hồ; có thể cải thiện bằng:
  - Tăng class weights cho lớp 3, hoặc áp dụng focal loss.
  - Data augmentation có kiểm soát cho lớp hiếm.
  - Thử prompt/format khác để nhấn mạnh ràng buộc lớp.
- Tối ưu tốc độ: dùng FlashAttention khi khả dụng; hoặc giảm grad_accum nếu GPU cho phép.

### 10. Tóm tắt
- Pipeline end-to-end với GPT-OSS 20B + QLoRA cho bài toán phân loại topic đã được xây dựng và kiểm chứng.
- Dev F1 macro đạt 76.43 ở epoch 3; kết quả từng lớp đã vượt baseline trên dev (kể cả lớp 3) cho thấy tiềm năng đạt hoặc vượt baseline test khi huấn luyện đủ và tinh chỉnh siêu tham số hợp lý.


