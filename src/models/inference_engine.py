"""
Inference engine for human-value classification on ParlaMint arguments.

Loads a fine-tuned RoBERTa classifier, scores each argument, aggregates
per-politician value frequencies, and renders triple-level Schwartz radar
charts (specific / basic / higher-order) to the reports/ folder.
"""

import os
os.environ.setdefault("HF_HOME", "/home/alumno/Desktop/datos/hf_cache")
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

import json
import re
from math import pi

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

from models.transformers import UniversalTransformerClassifier


# ==========================================
# Schwartz value taxonomy
# ==========================================
SPECIFIC_VALUES = [
    "Self-direction: thought", "Self-direction: action", "Stimulation",
    "Hedonism", "Achievement", "Power: dominance", "Power: resources",
    "Face", "Security: personal", "Security: societal", "Tradition",
    "Conformity: rules", "Conformity: interpersonal", "Humility",
    "Benevolence: caring", "Benevolence: dependability",
    "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance", "Universalism: objectivity",
]

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
    "Openness to Change": [
        "Self-direction: thought", "Self-direction: action", "Stimulation",
    ],
    "Self-Enhancement": [
        "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face",
    ],
    "Conservation": [
        "Security: personal", "Security: societal", "Tradition",
        "Conformity: rules", "Conformity: interpersonal", "Humility",
    ],
    "Self-Transcendence": [
        "Benevolence: caring", "Benevolence: dependability",
        "Universalism: concern", "Universalism: nature",
        "Universalism: tolerance", "Universalism: objectivity",
    ],
}


# ==========================================
# Engine
# ==========================================
class InferenceEngine:
    def __init__(
        self,
        model_name: str = "roberta-base",
        weights_path: str = "models/saved_weights/roberta_best.pt",
        reports_dir: str = "reports",
        threshold: float = 0.49,
        max_len: int = 128,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.weights_path = weights_path
        self.reports_dir = reports_dir
        self.threshold = threshold
        self.max_len = max_len
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        os.makedirs(self.reports_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.results_df = None

    # ---------- model ----------

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = UniversalTransformerClassifier(
            model_name=self.model_name, n_classes=len(SPECIFIC_VALUES)
        ).to(self.device)

        if os.path.exists(self.weights_path):
            state = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"✅ Loaded fine-tuned weights from {self.weights_path} "
                  f"on {self.device}")
        else:
            print(f"⚠️  No weights found at {self.weights_path} — "
                  f"using base {self.model_name} (untrained head).")

        self.model.eval()
        return self

    # ---------- data ----------

    @staticmethod
    def load_dataset(jsonl_path: str) -> pd.DataFrame:
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        df["text_raw"] = (
            df["Conclusion"] + " [SEP] " + df["Stance"] + " [SEP] " + df["Premise"]
        )
        print(f"✅ Loaded {len(df)} arguments from {jsonl_path}")
        return df

    # ---------- inference ----------

    @torch.inference_mode()
    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        if self.model is None:
            self.load_model()

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
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

        probs = np.vstack(all_probs)
        values = [
            [SPECIFIC_VALUES[i] for i in np.where(row > self.threshold)[0]]
            for row in probs
        ]

        self.results_df = pd.DataFrame({
            "Speaker": df["Speaker"].values,
            "Party": df.get("Party", pd.Series([""] * len(df))).values,
            "Values": values,
        })
        return self.results_df

    # ---------- aggregation ----------

    @staticmethod
    def _frequencies(values_series, mapping=None, specific_list=None):
        total = len(values_series)
        freqs = {}
        if mapping:
            for cat, subs in mapping.items():
                count = sum(
                    1 for vals in values_series if any(v in vals for v in subs)
                )
                freqs[cat] = 100 * count / total
        else:
            for v in specific_list:
                count = sum(1 for vals in values_series if v in vals)
                freqs[v] = 100 * count / total
        return freqs

    def aggregate(self, speaker: str) -> dict:
        sub = self.results_df[self.results_df["Speaker"] == speaker]
        if sub.empty:
            raise ValueError(f"No arguments found for speaker '{speaker}'")
        vals = sub["Values"]
        return {
            "n_arguments": len(sub),
            "higher": self._frequencies(vals, mapping=HIGHER_ORDER_MAP),
            "basic": self._frequencies(vals, mapping=BASIC_MAP),
            "specific": self._frequencies(vals, specific_list=SPECIFIC_VALUES),
        }

    # ---------- plotting ----------

    @staticmethod
    def _plot_radar(ax, data_dict, title, color):
        cats = list(data_dict.keys())
        vals = list(data_dict.values())
        n = len(cats)
        angles = [i / n * 2 * pi for i in range(n)]
        vals += vals[:1]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, fontsize=9, wrap=True)
        ax.set_rlabel_position(0)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="grey", size=8)
        ax.set_ylim(0, 100)
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.3)
        ax.set_title(title, size=14, weight="bold", pad=20)

    def plot_speaker(self, speaker: str, save: bool = True, show: bool = False):
        agg = self.aggregate(speaker)

        fig, axes = plt.subplots(
            1, 3, figsize=(22, 7), subplot_kw=dict(polar=True)
        )
        fig.suptitle(
            f"Rhetorical Value Profile: {speaker}  (n={agg['n_arguments']})",
            fontsize=20, weight="bold", y=1.05,
        )

        self._plot_radar(axes[0], agg["higher"],
                         "Level 3: Higher-Order Values", "#1f77b4")
        self._plot_radar(axes[1], agg["basic"],
                         "Level 2: Basic Values", "#ff7f0e")
        self._plot_radar(axes[2], agg["specific"],
                         "Level 1: Specific Values", "#2ca02c")

        plt.tight_layout()

        path = None
        if save:
            slug = re.sub(r"[^\w]+", "_", speaker).strip("_").lower()
            path = os.path.join(self.reports_dir, f"{slug}_radar.png")
            fig.savefig(path, dpi=400, bbox_inches="tight")
            print(f"💾 Saved {path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return path

    # ---------- one-shot pipeline ----------

    def run(self, jsonl_path: str, batch_size: int = 32):
        """Load data, predict, and render one radar per distinct speaker."""
        df = self.load_dataset(jsonl_path)
        self.predict(df, batch_size=batch_size)

        paths = []
        for speaker in self.results_df["Speaker"].unique():
            paths.append(self.plot_speaker(speaker, save=True, show=False))

        print(f"\n🎉 Generated {len(paths)} radar chart(s) in {self.reports_dir}/")
        return self.results_df, paths
