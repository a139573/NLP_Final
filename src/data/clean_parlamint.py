"""
clean_parlamint.py

Cleans and standardizes the raw JSONL output from the LLM.
Now features aggressive metadata recovery.
"""

import json
import os
import re

# --- CONFIGURATION ---
INPUT_PATH = "data/processed/phase3_parlamint_framing.jsonl" # Ensure this points to your actual raw file
OUTPUT_PATH = "data/processed/parlamint_clean.jsonl"

LABEL_CORRECTIONS = {
    "Critica al Adversario (Polarización/Ataque personal)": "Crítica al Adversario (Polarización/Ataque personal)",
    "Crítica al Adversario": "Crítica al Adversario (Polarización/Ataque personal)",
    "Critica al Adversario": "Crítica al Adversario (Polarización/Ataque personal)",
    "Estado de Bienestar (Salud/Educación/Pensiones)": "Defensa del Estado de Bienestar (Salud/Educación/Pensiones)"
}

def normalize_text(text: str) -> str:
    if not text: return ""
    text = str(text).replace("\n", " ").replace("\r", "")
    return re.sub(r'\s+', ' ', text).strip()

def clean_labels(label_list: list) -> list:
    if not label_list: return []
    cleaned_labels = []
    for label in label_list:
        corrected_label = LABEL_CORRECTIONS.get(label, label)
        if corrected_label != "Transparencia y Participación Ciudadana":
            cleaned_labels.append(corrected_label)
    return list(set(cleaned_labels))

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Could not find {INPUT_PATH}")
        return

    print(f" Reading raw LLM extractions from: {INPUT_PATH}")
    
    cleaned_records = []
    dropped_empty = 0
    desconocido_count = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
                
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_labels = rec.get("value_labels", rec.get("Value_labels", []))
            valid_labels = clean_labels(raw_labels)
            
            if not valid_labels:
                dropped_empty += 1
                continue

            conclusion = normalize_text(rec.get("Conclusion", rec.get("conclusion", "")))
            stance = normalize_text(rec.get("Stance", rec.get("stance", "")))
            premise = normalize_text(rec.get("Premise", rec.get("premise", "")))
            text_raw = re.sub(r'\.\.+', '.', re.sub(r'\s+\.', '.', f"{conclusion}. {stance}. {premise}.")).strip(" .") + "."

            # --- AGGRESSIVE METADATA RECOVERY ---
            speaker = rec.get("Speaker", rec.get("speaker", rec.get("Nombre", "Desconocido")))
            party = rec.get("Party", rec.get("party", rec.get("Partido", "Desconocido")))
            date = rec.get("Date", rec.get("date", rec.get("Fecha", "Desconocido")))
            
            if speaker == "Desconocido":
                desconocido_count += 1

            cleaned_records.append({
                "Speaker": speaker,
                "Party": party,
                "Date": date,
                "text": text_raw,
                "labels": valid_labels
            })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for rec in cleaned_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"✅ Cleaned data saved to: {OUTPUT_PATH}")
    print(f" Retained {len(cleaned_records)} valid arguments. Dropped {dropped_empty} empty/invalid entries.")
    
    if desconocido_count > 0:
        print(f"⚠️  WARNING: {desconocido_count} arguments are STILL 'Desconocido'. The LLM completely dropped the metadata.")
    else:
        print(f" SUCCESS: 0 'Desconocido' arguments! Metadata fully recovered.")

if __name__ == "__main__":
    main()