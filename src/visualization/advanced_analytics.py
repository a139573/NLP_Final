"""
Advanced Analytics Generator for Political Discourse.
Focus: 1D Ideological Gradient and Ranked Framing.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Add project root to path so we can import the inference engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.visualization.inference_engine import InferenceEngine, PHASE_DEFAULTS

# ==========================================
# CONFIGURATION & AESTHETICS
# ==========================================
PARTY_COLORS = {
    "PSOE": "#E30613",       # Red
    "PP": "#0155A4",         # Blue
    "Vox": "#63BE21",        # Green
    "Cs": "#EB6109",         # Orange
    "Podemos": "#672F6C",    # Purple
    "Desconocido": "#808080" # Grey (Baseline)
}

# ==========================================
# 1D GRADIENT WEIGHTS (Left vs. Right)
# ==========================================
# Positive values pull Right, Negative values pull Left
IDEOLOGY_WEIGHTS = {
    "Libertad Económica y Mercado": 1.5,
    "Unidad Nacional y Soberanía": 1.0,
    "Tradición y Valores Morales": 1.0,
    "Ley, Orden y Seguridad Institucional": 0.5,
    "Defensa del Estado de Bienestar (Salud/Educación/Pensiones)": -1.2,
    "Justicia Social e Igualdad": -1.5,
    "Feminismo y Derechos Civiles": -1.0,
    "Protección del Medio Ambiente": -0.8
}

def get_speaker_party(df, speaker):
    parties = df[df['Speaker'] == speaker]['Party'].unique()
    for p in parties:
        if p in PARTY_COLORS:
            return p
    return "Desconocido"

def plot_ideological_gradient(df, output_path):
    stats = []
    speakers = df['Speaker'].unique()
    
    for speaker in speakers:
        sub = df[df['Speaker'] == speaker]
        if len(sub) < 10: continue
            
        score = 0.0
        for frame, weight in IDEOLOGY_WEIGHTS.items():
            pct = sum(1 for labels in sub['Predicted_Labels'] if frame in labels) / len(sub)
            score += (pct * 100) * weight
            
        party = get_speaker_party(df, speaker)
        display_name = "Baseline" if speaker == "Desconocido" else speaker.split(",")[0]
        stats.append({'Speaker': display_name, 'Score': score, 'Party': party})

    stats_df = pd.DataFrame(stats).sort_values(by='Score')

    plt.figure(figsize=(12, 4))
    sns.set_theme(style="white")
    
    plt.axvspan(-100, 0, color='red', alpha=0.1, zorder=0)
    plt.axvspan(0, 100, color='blue', alpha=0.1, zorder=0)
    plt.axvline(0, color='black', lw=2, linestyle='-', zorder=1)

    for i, row in stats_df.iterrows():
        color = PARTY_COLORS.get(row['Party'], "#808080")
        plt.scatter(row['Score'], 0, color=color, s=400, edgecolors='black', zorder=5)
        
        y_offset = 0.05 if i % 2 == 0 else -0.05
        plt.text(row['Score'], y_offset, row['Speaker'], 
                 fontsize=12, fontweight='bold', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    plt.title("Rhetorical Ideological Spectrum (Left to Right)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("← Left-Wing Rhetoric                      Right-Wing Rhetoric →", fontsize=12, fontweight='bold')
    
    plt.yticks([])
    plt.ylim(-0.15, 0.15)
    
    limit = max(stats_df['Score'].abs().max() * 1.2, 50)
    plt.xlim(-limit, limit)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"🌈 Saved 1D Ideological Gradient: {output_path}")

def plot_ranked_frame(df, target_frame, output_path, title):
    stats = []
    for speaker in df['Speaker'].unique():
        sub = df[df['Speaker'] == speaker]
        if len(sub) < 10: continue
            
        count = sum(1 for labels in sub['Predicted_Labels'] if target_frame in labels)
        pct = (count / len(sub)) * 100
        party = get_speaker_party(df, speaker)
        display_name = "Baseline" if speaker == "Desconocido" else speaker.split(",")[0]
        stats.append({'Speaker': display_name, 'Percentage': pct, 'Party': party})
        
    stats_df = pd.DataFrame(stats).sort_values(by='Percentage', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    palette = {row['Speaker']: PARTY_COLORS.get(row['Party'], "#808080") for _, row in stats_df.iterrows()}
    
    ax = sns.barplot(x='Percentage', y='Speaker', data=stats_df, palette=palette)
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Percentage of Arguments (%)', fontsize=12)
    plt.ylabel('')
    plt.xlim(0, max(stats_df['Percentage']) + 10)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 0.5, p.get_y() + p.get_height() / 2.), va='center', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Ranked Bar Chart: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate advanced analytical visualizations.")
    parser.add_argument("--phase", type=int, default=3, help="Pipeline phase (default: 3).")
    parser.add_argument("--reports-dir", type=str, default="reports/figures")
    args = parser.parse_args()

    defaults = PHASE_DEFAULTS[args.phase]
    engine = InferenceEngine(model_dir=defaults["model_dir"], threshold=0.5)
    
    print("🧠 Running inference for analytics...")
    df = engine.load_dataset(defaults["data_path"])
    results_df = engine.predict(df, batch_size=256)
    
    os.makedirs(args.reports_dir, exist_ok=True)
    
    plot_ideological_gradient(results_df, f"{args.reports_dir}/ideology_gradient.png")
    
    plot_ranked_frame(
        results_df, 
        target_frame="Crítica al Adversario (Polarización/Ataque personal)", 
        output_path=f"{args.reports_dir}/ranking_critica_adversario.png", 
        title="Frequency of Oppositional Framing ('Crítica al Adversario')"
    )
    
    print("\n✨ Analytics generated successfully!")

if __name__ == "__main__":
    main()