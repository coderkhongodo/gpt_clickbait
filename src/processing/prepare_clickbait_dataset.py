import os
import json
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split


# Dataset paths
DATASET_CSV = os.environ.get("CLICKBAIT_CSV", os.path.join("data", "data_raw", "clickbait_dataset_vietnamese.csv"))
DATASET_JSONL = os.environ.get("CLICKBAIT_JSONL", os.path.join("data", "data_raw", "clickbait_dataset_vietnamese.jsonl"))

# Output directory for clickbait classification instructions
OUTPUT_DIR = os.environ.get("DATA_DIR", os.path.join("data_processed", "jsonl_text_clickbait"))


def load_clickbait_dataset(use_csv: bool = True) -> pd.DataFrame:
    """Load clickbait dataset from CSV or JSONL format"""
    if use_csv and os.path.exists(DATASET_CSV):
        print(f"Loading dataset from CSV: {DATASET_CSV}")
        df = pd.read_csv(DATASET_CSV)
    elif os.path.exists(DATASET_JSONL):
        print(f"Loading dataset from JSONL: {DATASET_JSONL}")
        df = pd.read_json(DATASET_JSONL, lines=True)
    else:
        raise FileNotFoundError(f"Neither {DATASET_CSV} nor {DATASET_JSONL} found")
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def preprocess_clickbait_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the clickbait dataset"""
    # Remove rows with missing essential data
    df = df.dropna(subset=['title', 'label'])
    
    # Fill missing lead_paragraph with empty string
    df['lead_paragraph'] = df['lead_paragraph'].fillna('')
    
    # Map labels to binary format: clickbait=1, non-clickbait=0
    label_mapping = {'clickbait': '1', 'non-clickbait': '0'}
    df['binary_label'] = df['label'].map(label_mapping)
    
    # Remove any rows with unmapped labels
    df = df.dropna(subset=['binary_label'])
    
    print(f"After preprocessing: {len(df)} samples")
    print(f"Binary label distribution:\n{df['binary_label'].value_counts()}")
    
    return df


def create_text_input(row: pd.Series) -> str:
    """Create text input from title only (lead paragraph is ignored)"""
    title = row['title'].strip()
    return f"Tiêu đề: {title}"


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/validation/test sets with stratification"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state, 
        stratify=df['binary_label']
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1 - val_size), 
        random_state=random_state, 
        stratify=temp_df['binary_label']
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples") 
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def convert_to_instruction_format(df: pd.DataFrame) -> List[dict]:
    """Convert dataframe to instruction format for training"""
    records = []
    
    # Instruction prompt for clickbait detection (title-only)
    instruction = (
        "Bạn là bộ phân loại clickbait.\n"
        "Nhiệm vụ: Xác định xem tiêu đề bài báo có phải là clickbait hay không.\n"
        "Quy ước:\n"
        "- 0 = không phải clickbait: tiêu đề trung thực, không câu view\n"
        "- 1 = clickbait: tiêu đề câu view, phóng đại, gây tò mò để thu hút click\n\n"
        "Chỉ trả đúng MỘT ký tự trong {0,1}. Không thêm chữ nào khác.\n\n"
        "Tiêu đề cần phân loại:"
    )
    
    for _, row in df.iterrows():
        text_input = create_text_input(row)
        
        records.append({
            "instruction": instruction,
            "input": f"{text_input}\nĐáp án:",
            "output": row['binary_label']
        })
    
    return records


def ensure_output_dir(path: str) -> None:
    """Create output directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def write_jsonl(records: List[dict], out_path: str) -> None:
    """Write records to JSONL file"""
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def convert_split_to_jsonl(df: pd.DataFrame, split_name: str) -> str:
    """Convert a data split to instruction JSONL format"""
    records = convert_to_instruction_format(df)
    
    ensure_output_dir(OUTPUT_DIR)
    
    # Map split names to output filenames
    filename_mapping = {
        "train": "train_instruction.jsonl",
        "val": "val_instruction.jsonl", 
        "validation": "val_instruction.jsonl",
        "test": "test_instruction.jsonl"
    }
    
    filename = filename_mapping.get(split_name, f"{split_name}_instruction.jsonl")
    out_path = os.path.join(OUTPUT_DIR, filename)
    
    write_jsonl(records, out_path)
    print(f"Wrote {len(records)} records to {out_path}")
    
    return out_path


def main():
    """Main function to process clickbait dataset"""
    print("Processing clickbait dataset...")
    
    # Load and preprocess data
    df = load_clickbait_dataset(use_csv=True)
    df = preprocess_clickbait_data(df)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Convert each split to instruction format
    convert_split_to_jsonl(train_df, "train")
    convert_split_to_jsonl(val_df, "val") 
    convert_split_to_jsonl(test_df, "test")
    
    print(f"\nClickbait dataset processing complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files created:")
    print(f"  - train_instruction.jsonl ({len(train_df)} samples)")
    print(f"  - val_instruction.jsonl ({len(val_df)} samples)")
    print(f"  - test_instruction.jsonl ({len(test_df)} samples)")


if __name__ == "__main__":
    main()
