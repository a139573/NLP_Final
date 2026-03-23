"""
Run the fine-tuned RoBERTa model over ParlaMint arguments and attach
predicted human-value labels.

Input:  parlamint_arguments_dataset_ALL_YEARS.jsonl
Output: parlamint_arguments_with_values.jsonl
"""

import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/roberta-values-es"
INPUT_PATH = "data/processed/parlamint_arguments_dataset_ALL_YEARS.jsonl"
OUTPUT_PATH = "data/processed/parlamint_arguments_with_values.jsonl"

VALUE_LABELS = [
    "Self-direction: thought", "Self-direction: action", "Stimulation",
    "Hedonism", "Achievement", "Power: dominance", "Power: resources",
    "Face", "Security: personal", "Security: societal", "Tradition",
    "Conformity: rules", "Conformity: interpersonal", "Humility",
    "Benevolence: caring", "Benevolence: dependability",
    "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance", "Universalism: objectivity",
]


def build_text(rec: dict) -> str:
    """Match the [SEP]-joined format used at training time."""
    return f"{rec['Conclusion']} [SEP] {rec['Stance']} [SEP] {rec['Premise']}"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} arguments")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(records), args.batch_size)):
            batch = records[start:start + args.batch_size]
            texts = [build_text(r) for r in batch]

            enc = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()

            for rec, p in zip(batch, probs):
                rec["value_scores"] = {
                    lbl: round(float(p[i]), 4) for i, lbl in enumerate(VALUE_LABELS)
                }
                rec["value_labels"] = [
                    lbl for i, lbl in enumerate(VALUE_LABELS) if p[i] > args.threshold
                ]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
