import csv
from collections import defaultdict

def summarize_altruism_scores(csv_file: str):
    """
    Reads the experiment results CSV and computes the overall mean altruism score
    for each LLM model (based on the ALTRUISM_SCORE column).

    Args:
        csv_file (str): Path to the results CSV.

    Returns:
        dict: {llm_name: mean_altruism_score}
    """
    scores = defaultdict(list)

    # Read the CSV and collect scores
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            llm_name = row.get("llm_name", "").strip()
            score_str = row.get("ALTRUISM_SCORE", "").strip()
            if llm_name and score_str:
                try:
                    score = float(score_str)
                    scores[llm_name].append(score)
                except ValueError:
                    continue  # skip bad rows

    # Compute mean per LLM
    summary = {}
    for llm, vals in scores.items():
        if vals:
            summary[llm] = sum(vals) / len(vals)

    return summary


if __name__ == "__main__":
    # Example usage
    results_csv = "/Users/shadow33/Documents/Algoverse/code/arjun-jass/data/altruistic_gen_coalition_test_20250920_202442.csv"
    summary = summarize_altruism_scores(results_csv)

    print("=== Overall Altruism Scores by Model ===")
    for model, score in summary.items():
        print(f"{model:20s}: {score:.4f}")
