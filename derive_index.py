from helper.data import atomic_congestion_indexer, cost_sharing_indexer, dictator_indexer, social_context_indexer
from helper.data import prisonner_dilemma
from helper.data.non_atomic_indexer import NonAtomicIndexer
from helper.data.prisonner_dilemma import PrisonersDilemmaIndexer
from helper.data.social_context_indexer import SocialContextIndexer

import pandas as pd

# Dictionary to collect results across all indexers
results = {}

"""
# ----------------------------
# Non-Atomic Congestion Indexer
# ----------------------------
print("=== Non-Atomic Congestion Indexer ===")
non_atomic_indexer = NonAtomicIndexer(csv_file="data/SFT_nonatomiccongestion_results_20250921_134405.csv")
for llm, value in non_atomic_indexer.altruism.items():
    results.setdefault(llm, {})["Non-Atomic Congestion"] = value

# ----------------------------
# Social Context Indexer
# ----------------------------
print("=== Social Context Indexer ===")
social_context_indexer = SocialContextIndexer("data/SFT_socialcontext_results_20250921_132729.csv")
for llm, value in social_context_indexer.altruism.items():
    results.setdefault(llm, {})["Social Context"] = value
"""
# ----------------------------
# Dictator Game Indexer
# ----------------------------
print("=== Dictator Game Indexer ===")
dictator_indexer_obj = dictator_indexer.DictatorGameIndexer("data/SFT_dictatorgame_results_20250921_144349.csv")
for llm, value in dictator_indexer_obj.altruism.items():
    results.setdefault(llm, {})["Dictator Game"] = value
    
"""
# ----------------------------
# Atomic Congestion Indexer
# ----------------------------
print("=== Atomic Congestion Indexer ===")
atomic_congestion_indexer_obj = atomic_congestion_indexer.AtomicCongestionIndexer("data/SFT_atomiccongestion_results_20250921_125458.csv")
for llm, measures in atomic_congestion_indexer_obj.altruism.items():
    results.setdefault(llm, {})["Atomic Congestion"] = measures


# ----------------------------
# Cost Sharing Scheduler Indexer
# ----------------------------
print("=== Cost Sharing Scheduler Indexer ===")
cost_sharing_indexer_obj = cost_sharing_indexer.CostSharingSchedulerIndexer("data/SFT_cost_sharing_results_20250921_121716.csv")
for llm, value in cost_sharing_indexer_obj.altruism.items():
    results.setdefault(llm, {})["Cost Sharing"] = value


# ----------------------------
# Prisoner's Dilemma Indexer
# ----------------------------
print("=== Prisoner's Dilemma Indexer ===")
prisonner_dilemma_indexer_obj = PrisonersDilemmaIndexer("data/SFT_prisonersdilemma_results_20250921_123838.csv")
for llm, measures in prisonner_dilemma_indexer_obj.altruism.items():
    results.setdefault(llm, {})["Prisoner's Dilemma"] = measures

"""

# ----------------------------
# Convert Results to Table
# ----------------------------
# Build DataFrame (rows = metrics, columns = LLMs)
df = pd.DataFrame(results).T  # transpose so rows = LLMs
print("\n=== Aggregated Table ===")
print(df)

# ----------------------------
# Export to LaTeX Table
# ----------------------------

latex_table = df.to_latex(
    index=True,
    caption="Comparison of altruism-related indexes across LLMs and games.",
    label="tab:altruism_indexes",
    float_format="%.3f"
)

with open("altruism_indexes.tex", "w") as f:
    f.write(latex_table)

print("\nLaTeX table saved to altruism_indexes.tex")
