"""
clean_parlamint.py

Cleans and standardizes the raw JSONL output from the 72B LLM.
- Normalizes text (removes extra spaces, newlines)
- Concatenates arguments into a single 'text' field
- Fixes label typos and standardizes the taxonomy
- Drops interventions where the LLM failed to extract a frame
"""

import json
import os
import re

# --- CONFIGURATION ---
INPUT_PATH = "data/processed/parlamint_framing_ALL_YEARS.jsonl"
OUTPUT_PATH = "data/processed/parlamint_clean.jsonl"

# --- TYPO MAPPING ---
LABEL_CORRECTIONS = {
    "Critica al Adversario (Polarización/Ataque personal)": "Crítica al Adversario (Polarización/Ataque personal)",
    "Crítica al Adversario": "Crítica al Adversario (Polarización/Ataque personal)",
    "Critica al Adversario": "Crítica al Adversario (Polarización/Ataque personal)",
    "Estado de Bienestar (Salud/Educación/Pensiones)": "Defensa del Estado de Bienestar (Salud/Educación/Pensiones)"
}

def normalize_text(text: str) -> str:
    """Removes extra spaces and newlines."""
    if not text:
        return ""
    text = str(text).replace("\n", " ").replace("\r", "")
    return re.sub(r'\s+', ' ', text).strip()

def clean_labels(label_list: list) -> list:
    """Fixes typographical errors and standardizes labels."""
    if not label_list:
        return []
        
    cleaned_labels = []
    for label in label_list:
        corrected_label = LABEL_CORRECTIONS.get(label, label)
        
        # Remove any stray labels that don't belong in the Phase 3 taxonomy
        if corrected_label != "Transparencia y Participación Ciudadana":
            cleaned_labels.append(corrected_label)
            
    return list(set(cleaned_labels)) # Remove duplicates

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Could not find {INPUT_PATH}")
        return

    print(f"📖 Reading raw LLM extractions from: {INPUT_PATH}")
    
    cleaned_records = []
    dropped_empty = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1. Clean and validate labels
            raw_labels = rec.get("value_labels", [])
            valid_labels = clean_labels(raw_labels)
            
            # Drop arguments with no valid labels
            if not valid_labels:
                dropped_empty += 1
                continue

            # 2. Normalize text components
            conclusion = normalize_text(rec.get("Conclusion", ""))
            stance = normalize_text(rec.get("Stance", ""))
            premise = normalize_text(rec.get("Premise", ""))
            
            # 3. Build the final text string for the model
            text_raw = f"{conclusion}. {stance}. {premise}."
            # Clean up weird punctuation like ". ." if fields were empty
            text_raw = re.sub(r'\s+\.', '.', text_raw)
            text_raw = re.sub(r'\.\.+', '.', text_raw)

            # 4. Construct the clean record
            new_rec = {
                "Speaker": rec.get("Speaker", "Desconocido"),
                "Party": rec.get("Party", "Desconocido"),
                "Date": rec.get("Date", "Desconocido"),
                "text": text_raw,
                "labels": valid_labels
            }
            cleaned_records.append(new_rec)

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for rec in cleaned_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"✅ Cleaned data saved to: {OUTPUT_PATH}")
    print(f"📊 Retained {len(cleaned_records)} valid arguments. Dropped {dropped_empty} empty/invalid entries.")

if __name__ == "__main__":
    main()