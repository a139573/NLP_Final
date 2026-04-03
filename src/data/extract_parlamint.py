"""
Master script to extract political/moral arguments from ParlaMint-ES.
Supports three distinct phases of the NLP pipeline via the `--phase` argument.

Usage:
    # Phase 1: Touché format (No values/labels)
    python src/data/create_dataset.py --phase 1

    # Phase 2: Touché format (Schwartz 20 Human Values taxonomy)
    python src/data/create_dataset.py --phase 2

    # Phase 3: Touché format (10 Political Frames taxonomy)
    python src/data/create_dataset.py --phase 3 --n_interventions 100
"""

import os
os.environ.setdefault("HF_HOME", "/home/alumno/Desktop/datos/hf_cache")

import argparse
import csv
import glob
import json
import random
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
BASE_DIR = "data/raw/ParlaMint-ES/ParlaMint-ES.txt"
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4" # Defaulting to the 72B Heavyweight

TARGET_POLITICIANS = [
    "Sánchez Pérez-Castejón, Pedro",
    "Casado Blanco, Pablo",
    "Iglesias Turrión, Pablo",
    "Rivera Díaz, Albert",
    "Abascal Conde, Santiago",
    "Rajoy Brey, Mariano",
]

# ==========================================
# PHASE-SPECIFIC PROMPTS & CONFIGS
# ==========================================
PHASE_CONFIGS = {
    1: {
        "output": "data/processed/phase1_parlamint_nolabels.jsonl",
        "log": "reports/logs/processed_phase1.log",
        "prompt": """Eres un experto en ciencia política y minería de argumentación.
Tu tarea es analizar intervenciones parlamentarias y extraer argumentos que coincidan con el esquema del dataset ValueEval23.

Reglas estrictas de extracción:
1. "Conclusion": El objetivo o propuesta principal de la intervención. DEBE ser una afirmación normativa general en español que empiece por "Deberíamos..." o "El Estado debe...".
2. "Stance": Estrictamente "a favor" o "en contra" de la conclusión.
3. "Premise": Una cita literal, exacta y textual extraída de la intervención que justifique la postura. No resumas, extrae la subcadena exacta.
4. "value_labels": En esta fase, devuelve siempre un array vacío [].
5. Metadatos: Mantén "Speaker", "Party", "Date" intactos.

Salida:
- DEBES responder ÚNICAMENTE con un array JSON válido de objetos.
- Si no hay ningún argumento sustancial, devuelve un array vacío: [].
- Cero texto adicional."""
    },
    2: {
        "output": "data/processed/phase2_parlamint_schwartz.jsonl",
        "log": "reports/logs/processed_phase2.log",
        "prompt": """Eres un experto en ciencia política y en la Teoría de Valores Humanos Básicos de Schwartz.
Tu tarea es analizar intervenciones parlamentarias, extraer argumentos, y clasificar los valores morales a los que apelan.

TAXONOMÍA PERMITIDA (Solo puedes usar exactamente estos 20 strings):
[
    "Self-direction: thought", "Self-direction: action", "Stimulation",
    "Hedonism", "Achievement", "Power: dominance", "Power: resources",
    "Face", "Security: personal", "Security: societal", "Tradition",
    "Conformity: rules", "Conformity: interpersonal", "Humility",
    "Benevolence: caring", "Benevolence: dependability",
    "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance", "Universalism: objectivity"
]

Reglas estrictas de extracción y clasificación:
1. "Conclusion": El objetivo o propuesta principal de la intervención (Ej: "El Estado debe...").
2. "Stance": Estrictamente "a favor" o "en contra".
3. "Premise": Cita literal y textual de la intervención que justifica la postura.
4. "value_labels": Un array de strings con los valores de la TAXONOMÍA PERMITIDA que se defienden. Si no apela a ninguno, usa [].
5. Metadatos: Mantén "Speaker", "Party", "Date" intactos.

Salida:
- DEBES responder ÚNICAMENTE con un array JSON válido de objetos.
- Cero texto adicional."""
    },
    3: {
        "output": "data/processed/phase3_parlamint_framing.jsonl",
        "log": "reports/logs/processed_phase3.log",
        "prompt": """Eres un estricto analista de ciencia política evaluando el parlamento español.
Extrae los argumentos de la intervención y clasifica el "Marco Político" (Political Frame).

TAXONOMÍA PERMITIDA (Solo puedes usar exactamente estos 10 strings):
[
    "Justicia Social e Igualdad", 
    "Unidad Nacional y Soberanía", 
    "Libertad Económica y Mercado", 
    "Ley, Orden y Seguridad Institucional", 
    "Tradición y Valores Morales", 
    "Defensa del Estado de Bienestar (Salud/Educación/Pensiones)",
    "Regeneración y Lucha contra la Corrupción",
    "Feminismo y Derechos Civiles",
    "Protección del Medio Ambiente",
    "Crítica al Adversario (Polarización/Ataque personal)"
]

REGLAS CRÍTICAS:
1. "Conclusion": La tesis o ataque principal. (Ej: "Hay que bajar los impuestos").
2. "Stance": Estrictamente "a favor" o "en contra".
3. "Premise": Cita literal de la intervención que justifica la conclusión.
4. "value_labels": Un array con los marcos de la taxonomía. REGLA DE ORO: NO ASUMAS MARCOS. Solo asigna un marco si el político lo defiende o ataca EXPLÍCITAMENTE. Si es procedimental o vago, devuelve [].
5. Metadatos: Mantén "Speaker", "Party", "Date" intactos.

SALIDA:
- DEBES responder ÚNICAMENTE con un array JSON válido de objetos.
- Cero texto adicional."""
    }
}

def build_user_prompt(text, speaker, party, date):
    return f"""Metadatos:
- Speaker: {speaker}
- Party: {party}
- Date: {date}

Intervención:
"{text}"

JSON de salida:"""

# ==========================================
# BULLETPROOF JSON PARSER
# ==========================================
def parse_llm_response(response_text):
    """Aggressively hunts for the JSON array, ignoring LLM conversational text."""
    # Strategy 1: regex for the outermost [{...}] array
    match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Strategy 2: strip markdown fences
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
# EXTRACTOR (with randomized reading order)
# ==========================================
class ParlaMintExtractor:
    def __init__(self, base_dir, output_file, processed_log, model_id,
                 target_politicians, system_prompt, n_interventions=None, seed=42):
        self.base_dir = base_dir
        self.output_file = output_file
        self.processed_log = processed_log
        self.model_id = model_id
        self.target_politicians = target_politicians
        self.system_prompt = system_prompt
        self.n_interventions = n_interventions
        self.seed = seed

        self.model = None
        self.tokenizer = None
        self.processed_ids = self._load_processed_ids()

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    def _load_processed_ids(self):
        if not os.path.exists(self.processed_log):
            return set()
        with open(self.processed_log, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    # ------------------------------------------------------------------
    # Parse a single session file
    # ------------------------------------------------------------------
    def _parse_session(self, txt_file):
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

    # ------------------------------------------------------------------
    # Collect & SHUFFLE interventions
    # ------------------------------------------------------------------
    def collect_interventions(self):
        print(f"🔍 Buscando transcripciones en {self.base_dir}...")
        txt_files = sorted(glob.glob(
            os.path.join(self.base_dir, "**", "*.txt"), recursive=True
        ))
        if not txt_files:
            raise FileNotFoundError("No se encontraron archivos .txt.")

        interventions = []
        for txt_file in tqdm(txt_files, desc="Procesando archivos"):
            interventions.extend(self._parse_session(txt_file))

        # --- Filter already-processed ---
        before = len(interventions)
        interventions = [it for it in interventions if it["id"] not in self.processed_ids]
        if before != len(interventions):
            print(f"♻️  Omitiendo {before - len(interventions)} intervenciones ya procesadas.")

        # --- SHUFFLE so partial runs cover ALL years ---
        random.seed(self.seed)
        random.shuffle(interventions)
        print(f"🔀 Intervenciones aleatorizadas (seed={self.seed}).")

        # --- Optional cap ---
        if self.n_interventions is not None and interventions:
            interventions = interventions[: self.n_interventions]
            print(f"⚠️  Limitando a {len(interventions)} intervenciones pendientes.")

        # --- Show year distribution for transparency ---
        self._print_year_distribution(interventions)

        return interventions

    @staticmethod
    def _print_year_distribution(interventions):
        """Quick console summary of how many interventions fall in each year."""
        year_counts: dict[str, int] = {}
        for it in interventions:
            year = it["date"][:4] if len(it["date"]) >= 4 else "????"
            year_counts[year] = year_counts.get(year, 0) + 1
        if year_counts:
            print("📊 Distribución por año en cola de procesamiento:")
            for y in sorted(year_counts):
                print(f"    {y}: {year_counts[y]}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        print(f"🚀 Cargando {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("✅ Modelo cargado.\n")

    # ------------------------------------------------------------------
    # Single-item extraction
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _extract_one(self, item):
        prompt = build_user_prompt(item["text"], item["speaker"], item["party"], item["date"])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        gen = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        gen = gen[:, inputs.input_ids.shape[1]:]
        raw = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

        parsed = parse_llm_response(raw)
        return parsed if isinstance(parsed, list) else []

    # ------------------------------------------------------------------
    # Resume bookkeeping
    # ------------------------------------------------------------------
    def _mark_processed(self, intervention_id):
        self.processed_ids.add(intervention_id)
        os.makedirs(os.path.dirname(self.processed_log) or ".", exist_ok=True)
        with open(self.processed_log, "a", encoding="utf-8") as f:
            f.write(intervention_id + "\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        interventions = self.collect_interventions()
        if not interventions:
            print("✅ Todo extraído.")
            return

        self.load_model()
        n_args = 0
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, "a", encoding="utf-8") as f_out:
            for item in tqdm(interventions, desc="Extrayendo argumentos"):
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
                    print(f"\n❌ Error ({item['id']}): {e}")

        print(f"\n🎉 Listo. {n_args} argumentos guardados en {self.output_file}.")


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract political arguments from ParlaMint-ES (multi-phase, randomized order)."
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="Pipeline phase: 1=no labels, 2=Schwartz, 3=Political Frames")
    parser.add_argument("--n_interventions", type=int, default=None,
                        help="Max interventions to process (None = all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override default output path for the chosen phase")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help="Override the default model ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling interventions")
    args = parser.parse_args()

    cfg = PHASE_CONFIGS[args.phase]
    output = args.output or cfg["output"]
    log = cfg["log"]
    system_prompt = cfg["prompt"]

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Phase {args.phase} | Model: {args.model}")
    print(f"  Output: {output}")
    print(f"  Log:    {log}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    extractor = ParlaMintExtractor(
        base_dir=args.base_dir,
        output_file=output,
        processed_log=log,
        model_id=args.model,
        target_politicians=TARGET_POLITICIANS,
        system_prompt=system_prompt,
        n_interventions=args.n_interventions,
        seed=args.seed,
    )
    extractor.run()


if __name__ == "__main__":
    main()