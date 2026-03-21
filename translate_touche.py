"""
Merge Touche23 train/val/test CSVs and translate to Spanish using a local
Qwen2.5-14B-Instruct model (4-bit quantized). Supports resuming via checkpoints.

Output: Touche_Data/processed/touche_es.csv
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/alumno/Desktop/datos/hf_cache")

import re
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/home/alumno/Desktop/datos/hf_cache/hub"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

DATA_DIR = "Touche_Data/processed"
SPLITS = ["train.csv", "val.csv", "test.csv"]
CHECKPOINT_PATH = os.path.join(DATA_DIR, "touche_es_checkpoint.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "touche_es.csv")

STANCE_MAP = {
    "in favor of": "a favor",
    "against": "en contra",
}

SYSTEM_PROMPT = (
    "You are a professional English-to-Spanish translator. "
    "Translate the user's text into natural, fluent Spanish. "
    "Preserve the meaning exactly. Output ONLY the translation, "
    "with no preamble, quotes, or explanations."
)


def load_merged() -> pd.DataFrame:
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(os.path.join(DATA_DIR, split))
        df["split"] = split.removesuffix(".csv")
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def parse_text_raw(text_raw: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in text_raw.split("[SEP]")]
    if len(parts) != 3:
        # Fallback: treat whole thing as premise
        return "", "in favor of", text_raw.strip()
    return parts[0], parts[1], parts[2]


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


@torch.inference_mode()
def translate_batch(texts: list[str], model, tokenizer, max_new_tokens: int = 256) -> list[str]:
    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": t},
        ]
        for t in texts
    ]

    prompts = tokenizer.apply_chat_template(
        messages_batch, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_only = out[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def build_text_clean(conclusion: str, premise: str) -> str:
    """Replicate the simple cleaning used in the original dataset."""
    joined = f"{conclusion} {premise}".lower()
    joined = re.sub(r"[^a-záéíóúüñ\s]", " ", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Checkpoint every N rows")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only translate the first N rows (dry-run)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path (disables checkpointing)")
    args = parser.parse_args()

    merged = load_merged()
    print(f"Merged dataset: {len(merged)} rows")

    if args.limit:
        merged = merged.head(args.limit).reset_index(drop=True)
        print(f"Limited to first {len(merged)} rows")

    global CHECKPOINT_PATH, OUTPUT_PATH
    if args.output:
        OUTPUT_PATH = args.output
        CHECKPOINT_PATH = args.output  # no separate checkpoint for dry-runs

    # Resume from checkpoint if present
    if os.path.exists(CHECKPOINT_PATH) and not args.output:
        ckpt = pd.read_csv(CHECKPOINT_PATH)
        done_ids = set(ckpt["Argument ID"].tolist())
        print(f"Resuming from checkpoint: {len(done_ids)} rows already translated")
    else:
        ckpt = pd.DataFrame()
        done_ids = set()

    todo = merged[~merged["Argument ID"].isin(done_ids)].reset_index(drop=True)
    if todo.empty:
        print("All rows already translated.")
        ckpt.to_csv(OUTPUT_PATH, index=False)
        return

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()

    label_cols = [
        c for c in merged.columns
        if c not in ("Argument ID", "text_raw", "text_clean", "split")
    ]

    new_rows = []
    pbar = tqdm(range(0, len(todo), args.batch_size), desc="Translating")

    for start in pbar:
        batch = todo.iloc[start:start + args.batch_size]

        parsed = [parse_text_raw(t) for t in batch["text_raw"]]
        conclusions = [p[0] for p in parsed]
        stances_en = [p[1] for p in parsed]
        premises = [p[2] for p in parsed]

        # Translate conclusions and premises in two batched passes
        concl_es = translate_batch(conclusions, model, tokenizer, max_new_tokens=128)
        prem_es = translate_batch(premises, model, tokenizer, max_new_tokens=256)

        for i, (_, row) in enumerate(batch.iterrows()):
            stance_es = STANCE_MAP.get(stances_en[i].lower().strip(), stances_en[i])
            text_raw_es = f"{concl_es[i]} [SEP] {stance_es} [SEP] {prem_es[i]}"

            out_row = {
                "Argument ID": row["Argument ID"],
                "text_raw": text_raw_es,
                "text_clean": build_text_clean(concl_es[i], prem_es[i]),
                "split": row["split"],
            }
            for col in label_cols:
                out_row[col] = row[col]
            new_rows.append(out_row)

        # Checkpoint
        processed = start + len(batch)
        if processed % args.save_every < args.batch_size or processed == len(todo):
            ckpt = pd.concat([ckpt, pd.DataFrame(new_rows)], ignore_index=True)
            ckpt.to_csv(CHECKPOINT_PATH, index=False)
            new_rows = []
            pbar.set_postfix(saved=len(ckpt))

    # Flush any remaining rows
    if new_rows:
        ckpt = pd.concat([ckpt, pd.DataFrame(new_rows)], ignore_index=True)
        ckpt.to_csv(CHECKPOINT_PATH, index=False)

    ckpt.to_csv(OUTPUT_PATH, index=False)
    print(f"Done. Wrote {len(ckpt)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
