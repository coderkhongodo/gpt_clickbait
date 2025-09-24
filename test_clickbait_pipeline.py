#!/usr/bin/env python3
"""
Test script to verify the clickbait detection pipeline works correctly.
This script tests data loading, preprocessing, and basic functionality.
"""

import os
import json
import sys
from pathlib import Path

def test_data_processing():
    """Test that clickbait data processing works correctly"""
    print("=" * 50)
    print("Testing Clickbait Data Processing")
    print("=" * 50)
    
    # Check if raw data exists
    raw_csv = Path("data/data_raw/clickbait_dataset_vietnamese.csv")
    raw_jsonl = Path("data/data_raw/clickbait_dataset_vietnamese.jsonl")
    
    if not (raw_csv.exists() or raw_jsonl.exists()):
        print("❌ ERROR: Raw clickbait dataset not found!")
        print(f"   Expected: {raw_csv} or {raw_jsonl}")
        return False
    
    print(f"✅ Raw data found: {raw_csv if raw_csv.exists() else raw_jsonl}")
    
    # Check if processed data exists
    processed_dir = Path("data_processed/jsonl_text_clickbait")
    train_file = processed_dir / "train_instruction.jsonl"
    val_file = processed_dir / "val_instruction.jsonl"
    test_file = processed_dir / "test_instruction.jsonl"
    
    if not all(f.exists() for f in [train_file, val_file, test_file]):
        print("❌ ERROR: Processed data not found!")
        print("   Run: python src/processing/prepare_clickbait_dataset.py")
        return False
    
    print("✅ Processed data files found")
    
    # Load and validate processed data
    try:
        for split, file_path in [("train", train_file), ("val", val_file), ("test", test_file)]:
            with open(file_path, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f]
            
            # Check record structure
            if not records:
                print(f"❌ ERROR: {split} file is empty!")
                return False

            sample = records[0]
            required_keys = ["instruction", "input", "output"]
            if not all(key in sample for key in required_keys):
                print(f"❌ ERROR: {split} records missing required keys!")
                print(f"   Expected: {required_keys}")
                print(f"   Found: {list(sample.keys())}")
                return False

            # Check that input format is title-only (no lead_paragraph content)
            sample_input = sample["input"]
            if "Nội dung:" in sample_input:
                print(f"❌ ERROR: {split} still contains lead_paragraph content!")
                print(f"   Sample input: {sample_input[:100]}...")
                return False

            if not sample_input.startswith("Tiêu đề:"):
                print(f"❌ ERROR: {split} input doesn't start with 'Tiêu đề:'!")
                print(f"   Sample input: {sample_input[:100]}...")
                return False

            # Check instruction format is updated for title-only
            sample_instruction = sample["instruction"]
            if "tiêu đề và nội dung bài báo" in sample_instruction:
                print(f"❌ ERROR: {split} instruction still mentions content!")
                return False

            if "tiêu đề bài báo" not in sample_instruction:
                print(f"❌ ERROR: {split} instruction not updated for title-only!")
                return False

            # Check label format
            labels = [r["output"] for r in records]
            valid_labels = {"0", "1"}
            invalid_labels = set(labels) - valid_labels
            if invalid_labels:
                print(f"❌ ERROR: {split} contains invalid labels: {invalid_labels}")
                return False

            label_counts = {label: labels.count(label) for label in valid_labels}
            print(f"✅ {split}: {len(records)} records, labels: {label_counts} (title-only format)")
    
    except Exception as e:
        print(f"❌ ERROR loading processed data: {e}")
        return False
    
    return True


def test_training_config():
    """Test that training configuration is updated for clickbait"""
    print("\n" + "=" * 50)
    print("Testing Training Configuration")
    print("=" * 50)
    
    train_script = Path("src/train/train_qlora_gpt_oss_20b.py")
    if not train_script.exists():
        print("❌ ERROR: Training script not found!")
        return False
    
    # Check if training script has been updated for clickbait
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for clickbait-specific configurations
    checks = [
        ("jsonl_text_clickbait", "Default data directory updated"),
        ("gpt-oss-20b-qlora-clickbait", "Default output directory updated"),
        ("1.0,1.0", "Binary class weights configured"),
        ("0.50,0.50", "Binary sampling ratio configured")
    ]
    
    for pattern, description in checks:
        if pattern in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - pattern '{pattern}' not found")
            return False
    
    return True


def test_evaluation_config():
    """Test that evaluation configuration is updated for clickbait"""
    print("\n" + "=" * 50)
    print("Testing Evaluation Configuration")
    print("=" * 50)
    
    eval_script = Path("src/eval/evaluate_model.py")
    if not eval_script.exists():
        print("❌ ERROR: Evaluation script not found!")
        return False
    
    # Check if evaluation script has been updated for clickbait
    with open(eval_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for clickbait-specific configurations
    checks = [
        ("jsonl_text_clickbait", "Test file path updated"),
        ("gpt-oss-20b-qlora-clickbait", "Default adapter directory updated"),
        ('"01"', "Binary labels configured")
    ]
    
    for pattern, description in checks:
        if pattern in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - pattern '{pattern}' not found")
            return False
    
    return True


def test_inference_config():
    """Test that inference configuration is updated for clickbait"""
    print("\n" + "=" * 50)
    print("Testing Inference Configuration")
    print("=" * 50)
    
    inference_script = Path("src/interface/inference_gpt_oss_20b.py")
    if not inference_script.exists():
        print("❌ ERROR: Inference script not found!")
        return False
    
    # Check if inference script has been updated for clickbait
    with open(inference_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for clickbait-specific configurations
    checks = [
        ("jsonl_text_clickbait", "Data directory updated"),
        ("gpt-oss-20b-qlora-clickbait", "Adapter directory updated"),
        ("test_instruction.jsonl", "Test file updated"),
        ("instruction", "Instruction format support")
    ]
    
    for pattern, description in checks:
        if pattern in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - pattern '{pattern}' not found")
            return False
    
    return True


def main():
    """Run all tests"""
    print("🚀 Testing Clickbait Detection Pipeline")
    print("=" * 60)
    
    tests = [
        ("Data Processing", test_data_processing),
        ("Training Config", test_training_config),
        ("Evaluation Config", test_evaluation_config),
        ("Inference Config", test_inference_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The clickbait pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Train the model: python src/train/train_qlora_gpt_oss_20b.py")
        print("2. Evaluate: python src/eval/evaluate_model.py")
        print("3. Run inference: python src/interface/inference_gpt_oss_20b.py")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
