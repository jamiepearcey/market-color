# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Embed a handful of causal sample questions for the A/B retrieval demo."""
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data"

# 20 BLIND emerging-markets analyst questions — framed from EM-desk concerns,
# NOT reverse-engineered from corpus headlines, so no single doc is guaranteed to
# self-explain them; assembling the answer across documents is the real task.
QUESTIONS = [
    "How is the US-Iran conflict affecting inflation and current-account balances in emerging-market oil importers?",
    "Which emerging-market currencies are most exposed to renewed US dollar strength?",
    "How are emerging-market central banks responding to imported energy inflation?",
    "What are the spillovers of Russia's fuel crisis onto Central Asian and former-Soviet economies?",
    "How is the Strait of Hormuz disruption reshaping how Asian emerging economies source their energy?",
    "Are foreign investors turning risk-off on emerging-market assets amid the Middle East conflict?",
    "How is China's weak domestic demand affecting other emerging-market exporters?",
    "What does the reopening of the Strait of Hormuz mean for emerging-market crude importers?",
    "How are commodity-exporting emerging markets faring as metals and oil prices swing?",
    "Is emerging-market and frontier sovereign debt under pressure from higher-for-longer US rates?",
    "How are Gulf economies positioning as oil prices normalize after the war?",
    "What is driving South Asian economies' scramble to secure energy imports?",
    "How is the war-driven inflation shock transmitting into emerging-market food and fuel prices?",
    "Which emerging markets stand to gain as the gold market's center of gravity shifts toward Asia?",
    "How are the oil-price swings affecting Latin American commodity economies?",
    "How is the global crypto-regulation wave affecting emerging-market digital-asset adoption?",
    "How is Kazakhstan balancing Russian and Iranian energy logistics routes?",
    "Are recent rate hikes in small open economies a response to the energy-inflation shock?",
    "How is India adjusting its crude-oil sourcing amid Middle East supply risk?",
    "What second-order effects is the European heatwave having on emerging-market manufacturers?",
]

from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
v = np.array(list(model.embed(QUESTIONS)), dtype=np.float32)
v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
np.save(OUT / "questions_custom.npy", v)
(OUT / "questions_custom.txt").write_text("\n".join(QUESTIONS) + "\n")
print(f"embedded {len(QUESTIONS)} questions -> {v.shape}")
