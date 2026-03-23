"""
Extract moral/political arguments from ParlaMint-ES parliamentary transcripts
into the ToucheEval23 format using a local Qwen2.5-14B-Instruct model.

Usage:
    # Full run (all interventions)
    python create_dataset.py

    # Toy run (first 30 interventions)
    python create_dataset.py --n_interventions 30 --output parlamint_arguments_dataset_TOY.jsonl

Already-processed intervention IDs (from processed_interventions.log) are
always skipped, so the script is resumable.
"""

import os
os.environ.setdefault("HF_HOME", "/home/alumno/Desktop/datos/hf_cache")

import argparse
import csv
import glob
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "data/raw/ParlaMint-ES/ParlaMint-ES.txt"
DEFAULT_OUTPUT = "data/processed/parlamint_arguments_dataset_ALL_YEARS.jsonl"
PROCESSED_LOG = "reports/processed_interventions.log"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

TARGET_POLITICIANS = [
    "Sánchez Pérez-Castejón, Pedro",
    "Casado Blanco, Pablo",
    "Iglesias Turrión, Pablo",
    "Rivera Díaz, Albert",
    "Abascal Conde, Santiago",
    "Rajoy Brey, Mariano",
]

# ==========================================
# LLM PROMPT TEMPLATES
# ==========================================
SYSTEM_PROMPT = """Eres un experto en ciencia política y minería de argumentación.
Tu tarea es analizar intervenciones parlamentarias y extraer argumentos morales y políticos que coincidan con el esquema del dataset ValueEval23.

Reglas estrictas de extracción:
1. "Conclusion": El objetivo o propuesta principal de la intervención. DEBE ser una afirmación normativa general en español que empiece por "Deberíamos..." o "El Estado debe...".
2. "Stance": Estrictamente "a favor" o "en contra" de la conclusión.
3. "Premise": Una cita literal, exacta y textual extraída de la intervención que justifique la postura. No resumas, extrae la subcadena exacta.
4. "Speaker", "Party", "Date": Usa exactamente los metadatos proporcionados.

Salida:
- DEBES responder ÚNICAMENTE con un array JSON válido de objetos.
- Si hay varios argumentos distintos, extrae varios objetos.
- Si no hay ningún argumento sustancial o moral (ej. solo saludos procesales), devuelve un array vacío: []
- No incluyas explicaciones, ni texto en markdown fuera del JSON.
"""


def build_user_prompt(text, speaker, party, date):
    return f"""Metadatos:
- Speaker: {speaker}
- Party: {party}
- Date: {date}

Intervención:
"{text}"

Extrae los argumentos en formato JSON:"""


def parse_llm_response(response_text):
    """Strip optional ```json fences and parse JSON; return None on failure."""
    clean = response_text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        return None


# ==========================================
# EXTRACTOR
# ==========================================
class ParlaMintExtractor:
    def __init__(self, base_dir, output_file, processed_log, model_id,
                 target_politicians, n_interventions=None):
        self.base_dir = base_dir
        self.output_file = output_file
        self.processed_log = processed_log
        self.model_id = model_id
        self.target_politicians = target_politicians
        self.n_interventions = n_interventions

        self.model = None
        self.tokenizer = None
        self.processed_ids = self._load_processed_ids()

    # ---------- corpus ----------

    def _load_processed_ids(self):
        if not os.path.exists(self.processed_log):
            return set()
        with open(self.processed_log, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _parse_session(self, txt_file):
        """Parse one transcript + its -meta.tsv sidecar into intervention dicts."""
        base = os.path.splitext(txt_file)[0]
        meta_file = base + "-meta.tsv"
        if not os.path.exists(meta_file) or meta_file.endswith("-meta-en.tsv"):
            return []

        try:
            txt_df = pd.read_csv(
                txt_file, sep="\t", header=None, names=["ID", "Text"],
                on_bad_lines="skip", quoting=csv.QUOTE_NONE,
            )
            meta_df = pd.read_csv(
                meta_file, sep="\t", on_bad_lines="skip", quoting=csv.QUOTE_NONE,
            )
        except Exception:
            return []

        txt_df["ID"] = txt_df["ID"].astype(str).str.strip()
        meta_df["ID"] = meta_df["ID"].astype(str).str.strip()
        meta_df["Speaker_name"] = meta_df["Speaker_name"].astype(str).str.strip()

        merged = pd.merge(txt_df, meta_df, on="ID", how="inner")
        merged = merged[merged["Speaker_name"].isin(self.target_politicians)]

        out = []
        for _, row in merged.iterrows():
            if pd.notna(row["Text"]):
                out.append({
                    "id": str(row["ID"]),
                    "text": str(row["Text"]),
                    "speaker": str(row["Speaker_name"]),
                    "party": str(row["Speaker_party"]),
                    "date": str(row["Date"]),
                })
        return out

    def collect_interventions(self):
        print(f"🔍 Scanning {self.base_dir} recursively for transcripts...")
        txt_files = sorted(
            glob.glob(os.path.join(self.base_dir, "**", "*.txt"), recursive=True)
        )
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found under {self.base_dir}. "
                "Ensure the ParlaMint-ES corpus is in the expected location."
            )

        interventions = []
        for txt_file in tqdm(txt_files, desc="Parsing files"):
            interventions.extend(self._parse_session(txt_file))

        print(f"✅ Found {len(interventions)} relevant interventions total.")

        # Skip already-processed IDs so we never fetch the same intervention twice
        before = len(interventions)
        interventions = [it for it in interventions if it["id"] not in self.processed_ids]
        if before != len(interventions):
            print(f"♻️  Skipping {before - len(interventions)} already-processed interventions.")

        if self.n_interventions is not None and interventions:
            interventions = interventions[: self.n_interventions]
            print(f"⚠️  Limiting to first {len(interventions)} pending interventions "
                  f"(--n_interventions {self.n_interventions}).")

        return interventions

    # ---------- model ----------

    def load_model(self):
        print(f"🚀 Loading {self.model_id} (bf16, device_map=auto)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        print("✅ Model loaded.\n")

    @torch.inference_mode()
    def _extract_one(self, item):
        """Run the LLM on one intervention, return list of argument dicts."""
        prompt = build_user_prompt(item["text"], item["speaker"], item["party"], item["date"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        gen = gen[:, inputs.input_ids.shape[1]:]
        raw = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

        parsed = parse_llm_response(raw)
        return parsed if isinstance(parsed, list) else []

    def _mark_processed(self, intervention_id):
        self.processed_ids.add(intervention_id)
        with open(self.processed_log, "a", encoding="utf-8") as f:
            f.write(intervention_id + "\n")

    # ---------- pipeline ----------

    def run(self):
        interventions = self.collect_interventions()
        if not interventions:
            print("✅ All interventions have already been extracted. "
                  "Nothing to do — no files were modified.")
            return

        self.load_model()

        n_args = 0
        with open(self.output_file, "a", encoding="utf-8") as f_out:
            for item in tqdm(interventions, desc="Extracting arguments"):
                if len(item["text"].strip()) < 20:
                    self._mark_processed(item["id"])
                    continue

                try:
                    for arg in self._extract_one(item):
                        f_out.write(json.dumps(arg, ensure_ascii=False) + "\n")
                        f_out.flush()
                        n_args += 1
                    self._mark_processed(item["id"])
                except Exception as e:
                    print(f"\n❌ Error for {item['speaker']} ({item['id']}): {e}")

        print(f"\n🎉 Done. Wrote {n_args} arguments to {self.output_file}.")


# ==========================================
# ENTRYPOINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_interventions", type=int, default=None,
        help="Process only the first N pending interventions (default: all).",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--base_dir", type=str, default=BASE_DIR,
        help="Root of the ParlaMint-ES .txt corpus.",
    )
    args = parser.parse_args()

    extractor = ParlaMintExtractor(
        base_dir=args.base_dir,
        output_file=args.output,
        processed_log=PROCESSED_LOG,
        model_id=MODEL_ID,
        target_politicians=TARGET_POLITICIANS,
        n_interventions=args.n_interventions,
    )
    extractor.run()


if __name__ == "__main__":
    main()
