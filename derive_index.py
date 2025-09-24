from helper.data import atomic_congestion_indexer, cost_sharing_indexer, dictator_indexer, social_context_indexer
from helper.data import prisonner_dilemma
from helper.data.non_atomic_indexer import NonAtomicIndexer
from helper.data.prisonner_dilemma import PrisonersDilemmaIndexer
from helper.data.social_context_indexer import SocialContextIndexer

import argparse
import glob
import os
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive altruism-related indexes from game CSV outputs."
    )

    # One optional file path per game. If omitted, use the existing defaults.
    parser.add_argument("--atomic", help="CSV for Atomic Congestion game")
    parser.add_argument("--nonatomic", help="CSV for Non-Atomic Congestion game")
    parser.add_argument("--social", help="CSV for Social Context game")
    parser.add_argument("--dictator", help="CSV for Dictator game")
    parser.add_argument("--cost", help="CSV for Cost Sharing game")
    parser.add_argument("--prisoner", help="CSV for Prisoner's Dilemma game")

    # Altruism-injected CSVs (per-game or combined)
    # These flags are repeatable; pass multiple files or glob patterns
    parser.add_argument("--altruistic_combined", action="append", help="CSV(s) from altruistic_injected_altruism_test_*.csv")
    parser.add_argument("--altruistic_hedonic", action="append", help="CSV(s) altruistic_injected_hedonicgame_results_*.csv")
    parser.add_argument("--altruistic_gencoalition", action="append", help="CSV(s) altruistic_injected_gencoalition_results_*.csv")
    parser.add_argument("--altruistic_dictator", action="append", help="CSV(s) altruistic_injected_dictatorgame_results_*.csv")
    parser.add_argument("--altruistic_prisoner", action="append", help="CSV(s) altruistic_injected_prisonersdilemma_results_*.csv")
    parser.add_argument("--altruistic_cost", action="append", help="CSV(s) altruistic_injected_costsharinggame_results_*.csv")
    parser.add_argument("--altruistic_atomic", action="append", help="CSV(s) altruistic_injected_atomiccongestion_results_*.csv")
    parser.add_argument("--altruistic_nonatomic", action="append", help="CSV(s) altruistic_injected_nonatomiccongestion_results_*.csv")
    parser.add_argument("--altruistic_social", action="append", help="CSV(s) altruistic_injected_socialcontext_results_*.csv")

    # Allow providing multiple files at once and skip missing ones
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip an indexer if the CSV path does not exist or fails to parse."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Dictionary to collect results across all indexers
    results: dict = {}

    # Defaults from previous hard-coded run (kept for convenience)
    default_nonatomic = "data/together_jaspertsh08_a03a_Qwen3_14B_Base_9cde0ab7_7630b2c7_SFT_nonatomiccongestion_results_20250923_221206.csv"
    default_social = "data/togetherai_SFT_socialcontext_results_20250921_195651.csv"
    default_dictator = "data/togetherai_SFT_dictatorgame_results_20250921_215546.csv"
    default_atomic = "data/togetherai_SFT_atomiccongestion_results_20250921_193047.csv"
    default_cost = "data/togetherai_SFT_costsharinggame_results_20250921_214443.csv"
    default_prisoner = "data/togetherai_SFT_prisonersdilemma_results_20250921_191812.csv"

    def run_or_skip(title: str, fn, *fn_args, **fn_kwargs):
        try:
            print(f"=== {title} ===")
            return fn(*fn_args, **fn_kwargs)
        except Exception as e:
            if args.skip_missing:
                print(f"[WARN] Skipping {title}: {e}")
                return None
            raise

    # ----------------------------
    # Non-Atomic Congestion Indexer
    # ----------------------------
    non_atomic_csv = args.nonatomic or default_nonatomic
    non_atomic_indexer = run_or_skip(
        "Non-Atomic Congestion Indexer",
        NonAtomicIndexer,
        csv_file=non_atomic_csv,
    )
    if non_atomic_indexer:
        for llm, value in non_atomic_indexer.altruism.items():
            results.setdefault(llm, {})["Non-Atomic Congestion"] = value

    # ----------------------------
    # Social Context Indexer
    # ----------------------------
    social_csv = args.social or default_social
    social_context_indexer_obj = run_or_skip(
        "Social Context Indexer",
        SocialContextIndexer,
        social_csv,
    )
    if social_context_indexer_obj:
        for llm, value in social_context_indexer_obj.altruism.items():
            results.setdefault(llm, {})["Social Context"] = value

    # ----------------------------
    # Dictator Game Indexer
    # ----------------------------
    dictator_csv = args.dictator or default_dictator
    dictator_indexer_obj = run_or_skip(
        "Dictator Game Indexer",
        dictator_indexer.DictatorGameIndexer,
        dictator_csv,
    )
    if dictator_indexer_obj:
        for llm, value in dictator_indexer_obj.altruism.items():
            results.setdefault(llm, {})["Dictator Game"] = value

    # ----------------------------
    # Atomic Congestion Indexer
    # ----------------------------
    atomic_csv = args.atomic or default_atomic
    atomic_congestion_indexer_obj = run_or_skip(
        "Atomic Congestion Indexer",
        atomic_congestion_indexer.AtomicCongestionIndexer,
        atomic_csv,
    )
    if atomic_congestion_indexer_obj:
        for llm, measures in atomic_congestion_indexer_obj.altruism.items():
            results.setdefault(llm, {})["Atomic Congestion"] = measures

    # ----------------------------
    # Cost Sharing Scheduler Indexer
    # ----------------------------
    cost_csv = args.cost or default_cost
    cost_sharing_indexer_obj = run_or_skip(
        "Cost Sharing Scheduler Indexer",
        cost_sharing_indexer.CostSharingSchedulerIndexer,
        cost_csv,
    )
    if cost_sharing_indexer_obj:
        for llm, value in cost_sharing_indexer_obj.altruism.items():
            results.setdefault(llm, {})["Cost Sharing"] = value

    # ----------------------------
    # Prisoner's Dilemma Indexer
    # ----------------------------
    prisoner_csv = args.prisoner or default_prisoner
    prisonner_dilemma_indexer_obj = run_or_skip(
        "Prisoner's Dilemma Indexer",
        PrisonersDilemmaIndexer,
        prisoner_csv,
    )
    if prisonner_dilemma_indexer_obj:
        for llm, measures in prisonner_dilemma_indexer_obj.altruism.items():
            results.setdefault(llm, {})["Prisoner's Dilemma"] = measures

    # ----------------------------
    # Altruism-injected CSVs handling
    # ----------------------------
    def _expand_many(paths_or_globs):
        files = []
        for p in (paths_or_globs or []):
            files.extend(glob.glob(p))
        # de-duplicate while preserving order
        seen = set()
        out = []
        for f in files:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def add_altruistic_csv(title: str, csv_paths):
        files = _expand_many(csv_paths)
        if not files:
            return
        frames = []
        for path in files:
            try:
                df = pd.read_csv(path)
                if "llm_name" not in df.columns or "ALTRUISM_SCORE" not in df.columns:
                    raise ValueError("Required columns 'llm_name' and 'ALTRUISM_SCORE' not found")
                frames.append(df[["llm_name", "ALTRUISM_SCORE"]])
                print(f"[OK] Loaded {path} for '{title}'")
            except Exception as e:
                if args.skip_missing:
                    print(f"[WARN] Skipping {path}: {e}")
                else:
                    raise
        if not frames:
            return
        merged = pd.concat(frames, ignore_index=True)
        series = merged.groupby("llm_name")["ALTRUISM_SCORE"].mean()
        for llm, val in series.items():
            results.setdefault(llm, {})[title] = float(val)
        print(f"Added altruistic index from {len(frames)} file(s) as '{title}'")

    # Combined altruistic file (already mixed game tasks, we report its mean as 'Altruism Injected (Combined)')
    add_altruistic_csv("Altruism Injected (Combined)", args.altruistic_combined)
    # Per-game altruistic files (each may aggregate multiple CSVs)
    add_altruistic_csv("Dictator (Altruistic)", args.altruistic_dictator)
    add_altruistic_csv("Prisoner's Dilemma (Altruistic)", args.altruistic_prisoner)
    add_altruistic_csv("Cost Sharing (Altruistic)", args.altruistic_cost)
    add_altruistic_csv("Atomic (Altruistic)", args.altruistic_atomic)
    add_altruistic_csv("NonAtomic (Altruistic)", args.altruistic_nonatomic)
    add_altruistic_csv("SocialContext (Altruistic)", args.altruistic_social)

    # ----------------------------
    # Convert Results to Table
    # ----------------------------
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
        float_format="%.3f",
    )

    with open("altruism_indexes.tex", "w") as f:
        f.write(latex_table)

    print("\nLaTeX table saved to altruism_indexes.tex")


if __name__ == "__main__":
    main()
