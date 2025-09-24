import pandas as pd

# Load LLM results and config
df = pd.read_csv("data/atomic_congestion_all.csv")
config_df = pd.read_csv("config/AtomicCongestion.csv")

# Ensure df is sorted by llm and round
df = df.sort_values(by=['llm', 'round']).reset_index(drop=True)

# Helper: get matrix string from a config row
def matrix_string(config_row):
    return f"R1R1:{config_row['R1R1']}; R1R2:{config_row['R1R2']}; R2R1:{config_row['R2R1']}; R2R2:{config_row['R2R2']}"

# Store cumulative times by LLM to calculate incremental travel time
last_cum_time = {}

matrix_strings = []

for idx, row in df.iterrows():
    llm = row['llm']
    cumulative_time = row['cumulative_time']

    # compute incremental travel time
    prev_cum = last_cum_time.get(llm, 0)
    incremental = cumulative_time - prev_cum
    last_cum_time[llm] = cumulative_time

    # find matching config row (customize matching logic if needed)
    # e.g., match by simulate_rounds or prompt
    config_row = config_df.iloc[0]  # fallback
    for _, c in config_df.iterrows():
        # if needed, match on simulate_rounds or prompt
        if c['total_rounds'] >= row['round']:
            config_row = c
            break

    # Now you could optionally verify incremental matches a matrix entry
    # For now, just attach full matrix string
    matrix_strings.append(matrix_string(config_row))

df['matrix'] = matrix_strings

# Save updated CSV
df.to_csv("results_with_matrix.csv", index=False)
print("Done! Saved to results_with_matrix.csv")
