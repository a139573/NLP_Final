"""
Data cleaning pipeline to remove LLM conversational artifacts, Chinese characters,
and semantic translation errors (Spanglish) from the quantized Qwen outputs.
"""

import os
import re
import pandas as pd

# Adjusted paths based on the project structure
INPUT_FILE = "data/interim/touche_es_checkpoint.csv"
OUTPUT_FILE = "data/interim/touche_es_clean.csv"

# Semantic dictionary for recurring LLM mistranslations
REPLACEMENTS = {
    "corta demasios corners": "toma demasiados atajos",
    "granjas factory": "granjas industriales",
    "la whaling": "la caza de ballenas",
    "La ballena blanca": "La caza de ballenas",
    "la ballena es inhumana": "la caza de ballenas es inhumana",
    "La ballena es cruel": "La caza de ballenas es cruel",
    "La ballena debería prohibirse": "La caza de ballenas debería prohibirse",
    "la agricultura granja industrial criadero intensivo": "la granja industrial",
    "la agricultura": "la granja industrial"
}

def clean_llm_artifacts(text_raw: str) -> str:
    """Removes conversational preambles, apologies, and Chinese notes from the LLM."""
    if not isinstance(text_raw, str):
        return text_raw

    # 1. Strip LLM preamble before the actual translation (Chinese and English versions)
    text_raw = re.sub(r'(?s)^.*?正确的翻译应该是：\s*', '', text_raw)
    text_raw = re.sub(r'(?s)^.*?here is the corrected version:\s*', '', text_raw)
    
    # 2. Extract text from quotes if the LLM wrapped its final translation in them
    text_raw = re.sub(r'(?s)^.*?请允许我完整提供翻译：“(.*?)(?:”|")\s*$', r'\1', text_raw)

    # 3. Strip any remaining Chinese characters (Unicode block)
    text_raw = re.sub(r'[\u4e00-\u9fff]', '', text_raw)
    
    # 4. Strip English (Note: ...) blocks
    text_raw = re.sub(r'\(Note:.*?\)', '', text_raw, flags=re.IGNORECASE | re.DOTALL)

    # 5. Apply semantic replacements
    for bad, good in REPLACEMENTS.items():
        text_raw = text_raw.replace(bad, good)
        # Handle Capitalized versions
        text_raw = text_raw.replace(bad.capitalize(), good.capitalize())

    # 6. Normalize ALL CAPS screaming text
    # Checks if the text has letters and all of them are uppercase
    if text_raw.isupper() and re.search(r'[A-Z]', text_raw):
        text_raw = text_raw.capitalize()

    # Clean up stray spaces/newlines left behind by regex replacements
    text_raw = re.sub(r'\s+', ' ', text_raw).strip()
    
    return text_raw

def build_text_clean(text_raw: str) -> str:
    """Rebuilds the text_clean column securely from the cleaned text_raw."""
    if not isinstance(text_raw, str):
        return ""
    # Remove the [SEP] tokens and apply standard cleaning
    text = text_raw.replace("[SEP]", " ")
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find {INPUT_FILE}")
        return

    print(f"Loading raw translations from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    initial_rows = len(df)
    
    print("Applying regex cleaning to 'text_raw'...")
    df["text_raw"] = df["text_raw"].apply(clean_llm_artifacts)
    
    print("Rebuilding 'text_clean' without LLM artifacts...")
    df["text_clean"] = df["text_raw"].apply(build_text_clean)
    
    # Optional: Drop rows that might still be critically malformed (missing SEP tokens)
    df_clean = df[df["text_raw"].str.count(r"\[SEP\]") == 2].copy()
    dropped = initial_rows - len(df_clean)
    
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved cleaned dataset to {OUTPUT_FILE}")
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} rows that were missing [SEP] tokens.")

if __name__ == "__main__":
    main()