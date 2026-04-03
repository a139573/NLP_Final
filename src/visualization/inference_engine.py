"""
Unified Inference Engine for Political Rhetoric Classification.

Automatically adapts to the Phase being tested based on the model's taxonomy:
- 20 Labels (Phase 1/2): Renders triple-level Schwartz value radar charts.
- 10 Labels (Phase 3): Renders a single political frame radar chart.
"""

import os
import json
import re
from math import pi

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# Schwartz Hierarchical Maps (Used only for Phase 1 & 2)
# ==========================================
BASIC_MAP = {
    "Self-direction": ["Self-direction: thought", "Self-direction: action"],
    "Stimulation": ["Stimulation"],
    "Hedonism": ["Hedonism"],
    "Achievement": ["Achievement"],
    "Power": ["Power: dominance", "Power: resources"],
    "Face": ["Face"],
    "Security": ["Security: personal", "Security: societal"],
    "Tradition": ["Tradition"],
    "Conformity": ["Conformity: rules", "Conformity: interpersonal"],
    "Humility": ["Humility"],
    "Benevolence": ["Benevolence: caring", "Benevolence: dependability"],
    "Universalism": [
        "Universalism: concern", "Universalism: nature",
        "Universalism: tolerance", "Universalism: objectivity",
    ],
}

HIGHER_ORDER_MAP = {
    "Openness to Change": ["Self-direction: thought", "Self-direction: action", "Stimulation"],
    "Self-Enhancement": ["Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face"],
    "Conservation": ["Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility"],
    "Self-Transcendence": ["Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance", "Universalism: objectivity"],
}


class InferenceEngine:
    def __init__(
        self,
        model_dir: str,
        reports_dir: str = "reports/figures",
        threshold: float = 0.85,
        max_len: int = 256,
        device: str | None = None,
    ):
        self.model_dir = model_dir
        self.reports_dir = reports_dir
        self.threshold = threshold
        self.max_len = max_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        os.makedirs(self.reports_dir, exist_ok=True)

        # Automatically load model, tokenizer, and taxonomy from the saved directory
        print(f"Loading model and taxonomy from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.num_classes = len(self.id2label)
        self.labels_list = [self.id2label[i] for i in range(self.num_classes)]
        
        print(f"✅ Model loaded successfully. Detected Phase Mode: {self.num_classes} classes.")
        self.results_df = None

    # ---------- data ----------
    # Removed @staticmethod so we can access self.num_classes
    def load_dataset(self, jsonl_path: str) -> pd.DataFrame:
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        
        # Safe concatenation (handling NaNs and matching training format)
        df["Conclusion"] = df["Conclusion"].fillna("").astype(str)
        df["Stance"] = df["Stance"].fillna("").astype(str)
        df["Premise"] = df["Premise"].fillna("").astype(str)
        
        # DYNAMIC FORMATTING BASED ON PHASE
        if self.num_classes == 20:
            # Phase 1 & 2 formatting
            df["text_raw"] = df["Conclusion"] + " [SEP] " + df["Stance"] + " [SEP] " + df["Premise"]
        else:
            # Phase 3 formatting
            df["text_raw"] = df["Conclusion"] + ". " + df["Stance"] + ". " + df["Premise"] + "."
        
        print(f"✅ Loaded {len(df)} arguments from {jsonl_path}")
        return df

    # ---------- inference ----------
    @torch.inference_mode()
    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        all_probs = []
        texts = df["text_raw"].tolist()

        for start in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            
            logits = self.model(**enc).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

        probs = np.vstack(all_probs)
        
        # Map probabilities > threshold to their text labels dynamically
        values = [
            [self.id2label[i] for i in np.where(row >= self.threshold)[0]]
            for row in probs
        ]

        self.results_df = pd.DataFrame({
            "Speaker": df.get("Speaker", pd.Series(["Unknown"] * len(df))).values,
            "Party": df.get("Party", pd.Series(["Unknown"] * len(df))).values,
            "Predicted_Labels": values,
        })
        return self.results_df

    # ---------- aggregation ----------
    def _frequencies(self, values_series, mapping=None, specific_list=None):
        total = len(values_series)
        if total == 0:
            return {}
            
        freqs = {}
        if mapping: # Used for Schwartz hierarchical mapping
            for cat, subs in mapping.items():
                count = sum(1 for vals in values_series if any(v in vals for v in subs))
                freqs[cat] = 100 * count / total
        else: # Used for flat lists (Specific Schwartz or Phase 3 Frames)
            for v in specific_list:
                count = sum(1 for vals in values_series if v in vals)
                freqs[v] = 100 * count / total
        return freqs

    def aggregate(self, speaker: str) -> dict:
        sub = self.results_df[self.results_df["Speaker"] == speaker]
        if sub.empty:
            raise ValueError(f"No arguments found for speaker '{speaker}'")
        vals = sub["Predicted_Labels"]
        
        if self.num_classes == 20:
            return {
                "n_arguments": len(sub),
                "higher": self._frequencies(vals, mapping=HIGHER_ORDER_MAP),
                "basic": self._frequencies(vals, mapping=BASIC_MAP),
                "specific": self._frequencies(vals, specific_list=self.labels_list),
            }
        else:
            # Phase 3: Flat taxonomy
            return {
                "n_arguments": len(sub),
                "frames": self._frequencies(vals, specific_list=self.labels_list)
            }

    # ---------- plotting ----------
    @staticmethod
    def _plot_radar(ax, data_dict, title, color, fill_color):
        cats = list(data_dict.keys())
        vals = list(data_dict.values())
        if not cats: return
        
        n = len(cats)
        angles = [i / n * 2 * pi for i in range(n)]
        vals += vals[:1]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        
        # Clean formatting for labels
        formatted_cats = [c.replace(" ", "\n") if len(c)>15 else c for c in cats]
        ax.set_xticklabels(formatted_cats, fontsize=9, wrap=True)
        
        ax.set_rlabel_position(0)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="grey", size=8)
        ax.set_ylim(0, 100)
        
        # Clean aesthetic palette
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=fill_color, alpha=0.35)
        ax.set_title(title, size=14, weight="bold", pad=20)
        ax.grid(color='#E0E0E0', linestyle='--', linewidth=0.5)

    def plot_speaker(self, speaker: str, save: bool = True, show: bool = False):
        agg = self.aggregate(speaker)
        slug = re.sub(r"[^\w]+", "_", speaker).strip("_").lower()

        if self.num_classes == 20:
            # Phase 1/2: Triple Radar
            fig, axes = plt.subplots(1, 3, figsize=(22, 7), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#F8F9FA')
            fig.suptitle(f"Rhetorical Profile (Schwartz): {speaker} (n={agg['n_arguments']})", fontsize=20, weight="bold", y=1.05)

            self._plot_radar(axes[0], agg["higher"], "Level 3: Higher-Order", "#2CA02C", "#2CA02C")
            self._plot_radar(axes[1], agg["basic"], "Level 2: Basic", "#1F77B4", "#1F77B4")
            self._plot_radar(axes[2], agg["specific"], "Level 1: Specific", "#FF7F0E", "#FF7F0E")
            
        else:
            # Phase 3: Single Radar
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#F8F9FA')
            fig.suptitle(f"Ideological Framing Profile: {speaker}\n(n={agg['n_arguments']} arguments)", fontsize=18, weight="bold", y=1.05)
            
            # Using the requested green/teal aesthetic
            self._plot_radar(ax, agg["frames"], "Political Frames", color="#2CA02C", fill_color="#4CAF50")

        plt.tight_layout()

        path = None
        if save:
            path = os.path.join(self.reports_dir, f"{slug}_phase_{self.num_classes}_radar.png")
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"💾 Saved {path}")

        if show: plt.show()
        else: plt.close(fig)

        return path

    def run(self, jsonl_path: str, batch_size: int = 32):
        df = self.load_dataset(jsonl_path) # <--- Make sure this looks like this
        self.predict(df, batch_size=batch_size)

        paths = []
        speakers = self.results_df["Speaker"].dropna().unique()
        for speaker in speakers:
            if str(speaker).lower() in ["nan", "unknown", ""]:
                continue
            
            # Generate radar only if speaker has > 5 arguments to avoid noisy charts
            if len(self.results_df[self.results_df["Speaker"] == speaker]) > 5:
                paths.append(self.plot_speaker(speaker, save=True, show=False))

        print(f"\n🎉 Generated {len(paths)} radar chart(s) in {self.reports_dir}/")
        return self.results_df, paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run taxonomy-agnostic classification and plot radars.")
    parser.add_argument("--data", type=str, default="data/processed/parlamint_framing_ALL_YEARS.jsonl")
    
    # --- UPDATE THIS LINE TO POINT TO THE _PRODUCTION FOLDER ---
    parser.add_argument("--model-dir", type=str, default="models/roberta-frames-optuna-FINAL_PRODUCTION") 
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reports-dir", type=str, default="reports/figures")
    
    args = parser.parse_args()

    engine = InferenceEngine(
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        threshold=args.threshold
    )
    engine.run(args.data, batch_size=args.batch_size)