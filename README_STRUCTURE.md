# Cấu trúc thư mục đề xuất

```
root/
├── data/
│   ├── sentiment/                  # Dữ liệu cho bài toán sentiment (instruction JSONL)
│   │   ├── train_instruction.jsonl
│   │   ├── val_instruction.jsonl
│   │   └── test_instruction.jsonl
│   └── topic/                      # Dữ liệu cho bài toán topic (instruction JSONL)
│       ├── train_instruction.jsonl
│       ├── val_instruction.jsonl
│       └── test_instruction.jsonl
│
├── tasks/
│   ├── sentiment/
│   │   ├── prepare_vsfc_sentiment.py  # Chuẩn bị dữ liệu sentiment (0/1)
│   │   ├── train.py                    # Gọi train_qlora... với biến môi trường phù hợp
│   │   ├── inference.py                # Suy luận nhanh 5 mẫu
│   │   └── eval.py                     # Đánh giá + classification report nhị phân
│   └── topic/
│       ├── prepare_vsfc_topic.py       # Chuẩn bị dữ liệu topic (0/1/2/3)
│       ├── train.py                    # Gọi train_qlora... với biến môi trường phù hợp
│       ├── inference.py                # Suy luận nhanh 5 mẫu
│       └── eval.py                     # Đánh giá + classification report 4 lớp
│
├── models/
│   ├── sentiment/                   # Nơi lưu adapter đã fine-tune cho sentiment
│   └── topic/                       # Nơi lưu adapter đã fine-tune cho topic
│
├── uit-vsfc/                        # Bộ dữ liệu gốc (sents.txt/sentiments.txt/topics.txt)
│   └── ...
│
├── train_qlora_gpt_oss_20b.py       # Script train chung (dùng biến môi trường)
├── inference_gpt_oss_20b.py         # Script inference chung
├── eval_100_sentiment.py            # Eval nhị phân (sentiment)
├── eval_topic_4class.py             # Eval 4 lớp (topic)
├── README_semantic_sentiment.md     # Báo cáo sentiment
├── README_STRUCTURE.md              # File này
└── requirements.txt
```

Gợi ý dùng:
- Sentiment:
```bash
export DATA_DIR="data/sentiment"
export OUTPUT_DIR="models/sentiment/gpt-oss-20b-qlora-sent"
python3 tasks/sentiment/train.py
python3 tasks/sentiment/eval.py
```
- Topic:
```bash
export DATA_DIR="data/topic"
export OUTPUT_DIR="models/topic/gpt-oss-20b-qlora-topic"
python3 tasks/topic/train.py
python3 tasks/topic/eval.py
```
