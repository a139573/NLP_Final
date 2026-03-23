"""
Fine-tune a Spanish RoBERTa model for multi-label human-value classification
on the translated Touche23 dataset (touche_es.csv).

Pipeline only — run manually once touche_es.csv is ready.
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.data_utils import BertDataset

HF_CACHE = "/home/alumno/Desktop/datos/hf_cache/hub"
MODEL_ID = "dccuchile/bert-base-spanish-wwm-uncased"

DATA_PATH = "data/interim/touche_es_clean.csv"
OUTPUT_DIR = "models/bert-values-es"

VALUE_LABELS = [
    "Self-direction: thought", "Self-direction: action", "Stimulation",
    "Hedonism", "Achievement", "Power: dominance", "Power: resources",
    "Face", "Security: personal", "Security: societal", "Tradition",
    "Conformity: rules", "Conformity: interpersonal", "Humility",
    "Benevolence: caring", "Benevolence: dependability",
    "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance", "Universalism: objectivity",
]


def load_splits():
    df = pd.read_csv(DATA_PATH)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    
    # Fallback if checkpoint only contains train data currently
    if len(val) == 0 and len(test) == 0:
        val = train.sample(frac=0.1, random_state=42)
        train = train.drop(val.index)
        test = train.sample(frac=0.1, random_state=42)
        train = train.drop(test.index)
        
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    train_df, val_df, test_df = load_splits()
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)

    def make_ds(df):
        texts = df["text_raw"].fillna("")
        labels = df[VALUE_LABELS].values.astype(np.float32)
        return BertDataset(texts, labels, tokenizer, max_len=256)

    train_ds = make_ds(train_df)
    val_ds = make_ds(val_df)
    test_ds = make_ds(test_df)

    id2label = {i: l for i, l in enumerate(VALUE_LABELS)}
    label2id = {l: i for i, l in enumerate(VALUE_LABELS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        num_labels=len(VALUE_LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nTest set evaluation:")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
