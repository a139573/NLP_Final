"""
Build a tokenized HuggingFace DatasetDict from cleaned ParlaMint JSONL.

Pipeline position:
    clean_parlamint.py  →  build_features_roberta.py  →  train_engine.py

Inputs:
    data/processed/parlamint_clean.jsonl   (JSONL with 'text' and 'labels')

Outputs:
    data/training/hf_roberta_dataset/      (HF DatasetDict: train/val/test)
    data/training/taxonomia_clases.json    (sorted label list for id2label)

Usage:
    python src/features/build_features_roberta.py
    python src/features/build_features_roberta.py --input data/processed/custom.jsonl --max-len 512
"""

import os
os.environ.setdefault("HF_HOME", "/home/alumno/Desktop/datos/hf_cache")

import argparse
import json

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_INPUT = "data/processed/phase3_parlamint_clean.jsonl"
DEFAULT_OUTPUT = "data/training/phase3_hf_roberta_dataset"
DEFAULT_CLASSES = "data/training/taxonomia_clases.json"
MODEL_ID = "bertin-project/bertin-roberta-base-spanish"
DEFAULT_MAX_LEN = 256
SEED = 42


def load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame, dropping records with empty labels."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("labels"):
                records.append(rec)

    df = pd.DataFrame(records)
    print(f"📖 Loaded {len(df)} records from {path}")
    return df


def binarize_labels(df: pd.DataFrame, classes_path: str) -> tuple[np.ndarray, list[str]]:
    """Apply MultiLabelBinarizer and save the sorted class list."""
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(df["labels"])

    sorted_classes = sorted(mlb.classes_.tolist())
    mlb_sorted = MultiLabelBinarizer(classes=sorted_classes)
    label_matrix = mlb_sorted.fit_transform(df["labels"])

    os.makedirs(os.path.dirname(classes_path), exist_ok=True)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(sorted_classes, f, ensure_ascii=False, indent=4)
    print(f"💾 Saved {len(sorted_classes)} classes → {classes_path}")

    return label_matrix.astype(np.float32), sorted_classes


def stratified_split(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative stratification for multi-label train/val/test split.

    Falls back to random split if skmultilearn is unavailable.
    """
    try:
        from skmultilearn.model_selection import iterative_train_test_split

        indices = np.arange(len(df)).reshape(-1, 1)
        test_ratio = 1.0 - train_ratio - val_ratio

        # First split: train vs (val+test)
        train_idx, rest_idx, _, rest_labels = iterative_train_test_split(
            indices, label_matrix, test_size=(val_ratio + test_ratio)
        )
        # Second split: val vs test (proportional)
        val_fraction = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx, _, _ = iterative_train_test_split(
            rest_idx, rest_labels, test_size=(1.0 - val_fraction)
        )
        train_idx = train_idx.flatten()
        val_idx = val_idx.flatten()
        test_idx = test_idx.flatten()
        print("🔀 Stratified split via skmultilearn")

    except ImportError:
        print("⚠️  skmultilearn not found — falling back to random split")
        rng = np.random.RandomState(SEED)
        indices = rng.permutation(len(df))
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

    print(f"   Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


def tokenize_and_build(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_id: str,
    max_len: int,
    output_dir: str,
) -> DatasetDict:
    """Tokenize texts and assemble a HuggingFace DatasetDict."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def make_split(idx: np.ndarray) -> Dataset:
        texts = df.iloc[idx]["text"].tolist()
        labels = label_matrix[idx].tolist()

        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="np",
        )
        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        })

    ds = DatasetDict({
        "train": make_split(train_idx),
        "val": make_split(val_idx),
        "test": make_split(test_idx),
    })

    ds.save_to_disk(output_dir)
    print(f"✅ DatasetDict saved → {output_dir}")
    for split, d in ds.items():
        print(f"   {split}: {len(d)} rows")

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Build tokenized HF dataset from cleaned ParlaMint JSONL."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=str, default=DEFAULT_CLASSES)
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    args = parser.parse_args()

    df = load_jsonl(args.input)
    label_matrix, classes = binarize_labels(df, args.classes)
    train_idx, val_idx, test_idx = stratified_split(df, label_matrix)
    tokenize_and_build(
        df, label_matrix, train_idx, val_idx, test_idx,
        model_id=args.model_id, max_len=args.max_len,
        output_dir=args.output,
    )

    print(f"\n🎉 Feature engineering complete. {len(classes)} labels, {len(df)} samples.")


if __name__ == "__main__":
    main()
