import matplotlib.pyplot as plt
import pandas as pd

models = [
    "ChatGPT-4o",
    "GPT-3.5-T",
    "Gemini-2.5-F",
    "LLaMA-3.3",
    "Qwen3-14B",
    "Mixtral-8x7B",
]

# ---- Data from your "Results for Base Models" table ----
data = {
        "Non-Atomic Congestion": {
            "RHA": {
                "ChatGPT-4o": 0.423, "GPT-3.5-T": 0.429, "Gemini-2.5-F": 0.427,
                "LLaMA-3.3": 0.423, "Qwen3-14B": 0.502, "Mixtral-8x7B": 0.508
            },
            "MIR": {
                "ChatGPT-4o": 0.472, "GPT-3.5-T": 0.474, "Gemini-2.5-F": 0.475,
                "LLaMA-3.3": 0.472, "Qwen3-14B": 0.495, "Mixtral-8x7B": 0.510
            },
            "DSR": {
                "ChatGPT-4o": -16.200, "GPT-3.5-T": -16.510, "Gemini-2.5-F": -16.580,
                "LLaMA-3.3": -14.730, "Qwen3-14B": 0.971, "Mixtral-8x7B": 0.971
            },
        },
        "Social Context": {
            "Deviation": {
                "ChatGPT-4o": -6.503, "GPT-3.5-T": -6.005, "Gemini-2.5-F": -6.160,
                "LLaMA-3.3": -5.953, "Qwen3-14B": 1.478, "Mixtral-8x7B": 2.700
            },
            "Rank": {
                "ChatGPT-4o": 0.600, "GPT-3.5-T": 0.594, "Gemini-2.5-F": 0.588,
                "LLaMA-3.3": 0.580, "Qwen3-14B": 0.926, "Mixtral-8x7B": 0.714
            },
        },
        "Dictator Game": {
            "Util. Gain": {
                "ChatGPT-4o": -2.000, "GPT-3.5-T": 2.250, "Gemini-2.5-F": -0.125,
                "LLaMA-3.3": 2.167, "Qwen3-14B": 0.191, "Mixtral-8x7B": -0.472
            },
            "Warm-Glow": {
                "ChatGPT-4o": 33.430, "GPT-3.5-T": 43.860, "Gemini-2.5-F": 47.220,
                "LLaMA-3.3": 67.150, "Qwen3-14B": 40.010, "Mixtral-8x7B": 38.680
            },
        },
        "Atomic Congestion": {
            "Social Welfare": {
                "ChatGPT-4o": -34.400, "GPT-3.5-T": -34.770, "Gemini-2.5-F": -34.630,
                "LLaMA-3.3": -34.570, "Qwen3-14B": -6.470, "Mixtral-8x7B": -6.600
            },
            "SVO Angle": {
                "ChatGPT-4o": -2.281, "GPT-3.5-T": -2.387, "Gemini-2.5-F": -2.353,
                "LLaMA-3.3": -2.324, "Qwen3-14B": -2.309, "Mixtral-8x7B": -2.352
            },
        },
        "Cost Sharing": {
            "NCC": {
                "ChatGPT-4o": 1.070, "GPT-3.5-T": 1.072, "Gemini-2.5-F": 1.071,
                "LLaMA-3.3": 1.071, "Qwen3-14B": 1.058, "Mixtral-8x7B": 1.058
            },
            "FS Index": {
                "ChatGPT-4o": 0.055, "GPT-3.5-T": 0.056, "Gemini-2.5-F": 0.054,
                "LLaMA-3.3": 0.052, "Qwen3-14B": 0.067, "Mixtral-8x7B": 0.067
            },
        },
        "Prisoner's Dilemma": {
            "Cooperation Freq.": {
                "ChatGPT-4o": 0.449, "GPT-3.5-T": 0.461, "Gemini-2.5-F": 0.473,
                "LLaMA-3.3": 0.475, "Qwen3-14B": 0.408, "Mixtral-8x7B": 0.412
            },
            "Avg. Payoff": {
                "ChatGPT-4o": 0.536, "GPT-3.5-T": 0.449, "Gemini-2.5-F": 0.580,
                "LLaMA-3.3": 0.553, "Qwen3-14B": 0.647, "Mixtral-8x7B": 0.500
            },
            "MCS": {
                "ChatGPT-4o": 0.539, "GPT-3.5-T": 0.508, "Gemini-2.5-F": 0.464,
                "LLaMA-3.3": 0.525, "Qwen3-14B": 0.625, "Mixtral-8x7B": 0.592
            },
        },
        "Hedonic Game": {
            "Altruism Score": {
                "ChatGPT-4o": 0.095, "GPT-3.5-T": 0.085, "Gemini-2.5-F": 0.125,
                "LLaMA-3.3": 0.125, "Qwen3-14B": 0.054, "Mixtral-8x7B": 0.030
            }
        },
        "Coalition Game": {
            "Altruism Score": {
                "ChatGPT-4o": 0.642, "GPT-3.5-T": 0.812, "Gemini-2.5-F": 0.623,
                "LLaMA-3.3": 0.923, "Qwen3-14B": 0.772, "Mixtral-8x7B": 0.783
            }
        },
}

# Convert to DataFrames per game
dfs = {game: pd.DataFrame(metrics).T[models] for game, metrics in data.items()}

def plot_game(game_name: str, normalize: bool = False):
    """
    Create bar plots comparing models for each metric in the chosen game.
    
    Parameters
    ----------
    game_name : str
        Name of the game (must be a key in dfs).
    normalize : bool, optional
        If True, scales each metric's values between 0 and 1
        to highlight relative differences (useful when ranges differ a lot).
    """
    if game_name not in dfs:
        raise ValueError(f"Game '{game_name}' not found. Choose from: {list(dfs.keys())}")
    
    df = dfs[game_name]
    for metric in df.index:
        values = df.loc[metric]
        if normalize:
            # scale between 0 and 1 for visibility
            vals = (values - values.min()) / (values.max() - values.min() + 1e-9)
        else:
            vals = values

        plt.figure(figsize=(8, 5))
        bars = plt.bar(vals.index, vals.values, color="skyblue", edgecolor="black")
        plt.title(f"{game_name} — {metric}", fontsize=14)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(metric + (" (normalized)" if normalize else ""), fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # annotate each bar with actual value
        for bar, actual in zip(bars, values.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{actual:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()

# Example usage:
plot_game("Social Context")
# plot_game("Non-Atomic Congestion")
