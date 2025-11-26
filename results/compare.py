import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------
# Raw data from your three tables (Base / PI / SFT)
# ---------------------------------------------
MODELS = ["ChatGPT-4o", "GPT-3.5-T", "Gemini-2.5-F", "LLaMA-3.3", "Qwen3-14B", "Mixtral-8x7B"]

DATA = {
    "Base": {
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
            "Utility": {
                "ChatGPT-4o": 18409.320, "GPT-3.5-T": 18409.070, "Gemini-2.5-F": 18409.150,
                "LLaMA-3.3": 18409.050, "Qwen3-14B": 903.261, "Mixtral-8x7B": 902.650
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
    },

    "Prompt Injection": {
        "Non-Atomic Congestion": {
            "RHA": {
                "ChatGPT-4o": 0.289, "GPT-3.5-T": 0.296, "Gemini-2.5-F": 0.291,
                "LLaMA-3.3": 0.302, "Qwen3-14B": 0.400, "Mixtral-8x7B": 0.400
            },
            "MIR": {
                "ChatGPT-4o": 0.333, "GPT-3.5-T": 0.344, "Gemini-2.5-F": 0.330,
                "LLaMA-3.3": 0.342, "Qwen3-14B": 0.410, "Mixtral-8x7B": 0.411
            },
            "DSR": {
                "ChatGPT-4o": -12.090, "GPT-3.5-T": -11.650, "Gemini-2.5-F": -11.330,
                "LLaMA-3.3": -11.780, "Qwen3-14B": 0.979, "Mixtral-8x7B": 0.978
            },
        },
        "Social Context": {
            "Deviation": {
                "ChatGPT-4o": -5.664, "GPT-3.5-T": -5.300, "Gemini-2.5-F": -4.948,
                "LLaMA-3.3": -5.374, "Qwen3-14B": 3.552, "Mixtral-8x7B": 3.448
            },
            "Utility": {
                "ChatGPT-4o": 18653.930, "GPT-3.5-T": 18653.750, "Gemini-2.5-F": 18641.590,
                "LLaMA-3.3": 18653.790, "Qwen3-14B": 452.224, "Mixtral-8x7B": 452.276
            },
            "Rank": {
                "ChatGPT-4o": 0.455, "GPT-3.5-T": 0.452, "Gemini-2.5-F": 0.430,
                "LLaMA-3.3": 0.469, "Qwen3-14B": 0.098, "Mixtral-8x7B": 0.094
            },
        },
        "Dictator Game": {
            "Util. Gain": {
                "ChatGPT-4o": 1.750, "GPT-3.5-T": 0.333, "Gemini-2.5-F": 2.000,
                "LLaMA-3.3": 2.583, "Qwen3-14B": 0.870, "Mixtral-8x7B": 1.305
            },
            "Warm-Glow": {
                "ChatGPT-4o": 59.430, "GPT-3.5-T": 53.120, "Gemini-2.5-F": 54.630,
                "LLaMA-3.3": 69.640, "Qwen3-14B": 50.625, "Mixtral-8x7B": 50.494
            },
        },
        "Atomic Congestion": {
            "Social Welfare": {
                "ChatGPT-4o": -34.930, "GPT-3.5-T": -34.730, "Gemini-2.5-F": -34.800,
                "LLaMA-3.3": -34.670, "Qwen3-14B": -5.267, "Mixtral-8x7B": -5.233
            },
            "SVO Angle": {
                "ChatGPT-4o": -2.411, "GPT-3.5-T": -2.376, "Gemini-2.5-F": -2.395,
                "LLaMA-3.3": -2.347, "Qwen3-14B": -2.393, "Mixtral-8x7B": -2.320
            },
        },
        "Cost Sharing": {
            "NCC": {
                "ChatGPT-4o": 1.061, "GPT-3.5-T": 1.061, "Gemini-2.5-F": 1.061,
                "LLaMA-3.3": 1.061, "Qwen3-14B": 1.058, "Mixtral-8x7B": 1.058
            },
            "FS Index": {
                "ChatGPT-4o": 0.063, "GPT-3.5-T": 0.063, "Gemini-2.5-F": 0.063,
                "LLaMA-3.3": 0.063, "Qwen3-14B": 0.066, "Mixtral-8x7B": 0.066
            },
        },
        "Prisoner's Dilemma": {
            "Cooperation Freq.": {
                "ChatGPT-4o": 0.996, "GPT-3.5-T": 0.994, "Gemini-2.5-F": 0.984,
                "LLaMA-3.3": 0.992, "Qwen3-14B": 0.984, "Mixtral-8x7B": 0.990
            },
            "Avg. Payoff": {
                "ChatGPT-4o": 0.586, "GPT-3.5-T": 0.255, "Gemini-2.5-F": 0.479,
                "LLaMA-3.3": 0.491, "Qwen3-14B": 0.677, "Mixtral-8x7B": 0.527
            },
            "MCS": {
                "ChatGPT-4o": 0.996, "GPT-3.5-T": 1.000, "Gemini-2.5-F": 0.980,
                "LLaMA-3.3": 0.988, "Qwen3-14B": 0.996, "Mixtral-8x7B": 0.996
            },
        },
        "Hedonic Game": {
            "Altruism Score": {
                "ChatGPT-4o": 0.125, "GPT-3.5-T": 0.140, "Gemini-2.5-F": 0.125,
                "LLaMA-3.3": 0.125, "Qwen3-14B": 0.109, "Mixtral-8x7B": 0.109
            }
        },
        "Coalition Game": {
            "Altruism Score": {
                "ChatGPT-4o": 0.861, "GPT-3.5-T": 0.858, "Gemini-2.5-F": 0.858,
                "LLaMA-3.3": 0.861, "Qwen3-14B": 0.875, "Mixtral-8x7B": 0.848
            }
        },
    },

    "SFT": {
        "Non-Atomic Congestion": {
            "RHA": {
                "ChatGPT-4o": 0.149, "GPT-3.5-T": 0.342, "Gemini-2.5-F": 0.394,
                "LLaMA-3.3": 0.391, "Qwen3-14B": 0.454, "Mixtral-8x7B": 0.580
            },
            "MIR": {
                "ChatGPT-4o": 0.822, "GPT-3.5-T": 0.659, "Gemini-2.5-F": 0.427,
                "LLaMA-3.3": 0.410, "Qwen3-14B": 0.451, "Mixtral-8x7B": 0.528
            },
            "DSR": {
                "ChatGPT-4o": 0.992, "GPT-3.5-T": 0.982, "Gemini-2.5-F": 0.979,
                "LLaMA-3.3": 0.981, "Qwen3-14B": 0.974, "Mixtral-8x7B": 0.974
            },
        },
        "Social Context": {
            "Deviation": {
                "ChatGPT-4o": 4.422, "GPT-3.5-T": 2.578, "Gemini-2.5-F": 5.000,
                "LLaMA-3.3": 5.000, "Qwen3-14B": 1.822, "Mixtral-8x7B": 5.000
            },
            "Utility": {
                "ChatGPT-4o": 451.789, "GPT-3.5-T": 452.711, "Gemini-2.5-F": 151.500,
                "LLaMA-3.3": 151.500, "Qwen3-14B": 903.089, "Mixtral-8x7B": 151.500
            },
            "Rank": {
                "ChatGPT-4o": 0.010, "GPT-3.5-T": 0.630, "Gemini-2.5-F": 0.000,
                "LLaMA-3.3": 0.000, "Qwen3-14B": 0.785, "Mixtral-8x7B": 0.000
            },
        },
        "Dictator Game": {
            "Util. Gain": {
                "ChatGPT-4o": 1.239, "GPT-3.5-T": 1.152, "Gemini-2.5-F": 0.441,
                "LLaMA-3.3": -0.606, "Qwen3-14B": -0.074, "Mixtral-8x7B": 1.417
            },
            "Warm-Glow": {
                "ChatGPT-4o": 69.047, "GPT-3.5-T": 49.570, "Gemini-2.5-F": 43.443,
                "LLaMA-3.3": 46.404, "Qwen3-14B": 43.304, "Mixtral-8x7B": 45.637
            },
        },
        "Atomic Congestion": {
            "Social Welfare": {
                "ChatGPT-4o": -5.567, "GPT-3.5-T": -5.333, "Gemini-2.5-F": -192.503,
                "LLaMA-3.3": -3.600, "Qwen3-14B": -6.667, "Mixtral-8x7B": -3.267
            },
            "SVO Angle": {
                "ChatGPT-4o": -2.399, "GPT-3.5-T": -2.314, "Gemini-2.5-F": 0.000,
                "LLaMA-3.3": 0.000, "Qwen3-14B": -2.374, "Mixtral-8x7B": 0.000
            },
        },
        "Cost Sharing": {
            "NCC": {
                "ChatGPT-4o": 1.057, "GPT-3.5-T": 1.057, "Gemini-2.5-F": 1.056,
                "LLaMA-3.3": 1.057, "Qwen3-14B": 1.058, "Mixtral-8x7B": 1.053
            },
            "FS Index": {
                "ChatGPT-4o": 0.065, "GPT-3.5-T": 0.065, "Gemini-2.5-F": 0.069,
                "LLaMA-3.3": 0.067, "Qwen3-14B": 0.067, "Mixtral-8x7B": 0.063
            },
        },
        "Prisoner's Dilemma": {
            "Cooperation Freq.": {
                "ChatGPT-4o": 0.728, "GPT-3.5-T": 0.344, "Gemini-2.5-F": 0.248,
                "LLaMA-3.3": 0.228, "Qwen3-14B": 0.400, "Mixtral-8x7B": 0.876
            },
            "Avg. Payoff": {
                "ChatGPT-4o": 0.434, "GPT-3.5-T": 0.657, "Gemini-2.5-F": 0.573,
                "LLaMA-3.3": 0.447, "Qwen3-14B": 0.500, "Mixtral-8x7B": 0.370
            },
            "MCS": {
                "ChatGPT-4o": 0.862, "GPT-3.5-T": 0.716, "Gemini-2.5-F": 0.569,
                "LLaMA-3.3": 0.618, "Qwen3-14B": 0.670, "Mixtral-8x7B": 0.920
            },
        },
        "Hedonic Game": {
            "Altruism Score": {
                "ChatGPT-4o": 0.036, "GPT-3.5-T": 0.155, "Gemini-2.5-F": 0.065,
                "LLaMA-3.3": 0.024, "Qwen3-14B": 0.125, "Mixtral-8x7B": 0.125
            }
        },
        "Coalition Game": {
            "Altruism Score": {
                "ChatGPT-4o": 0.962, "GPT-3.5-T": 0.948, "Gemini-2.5-F": 0.9158,
                "LLaMA-3.3": 0.943, "Qwen3-14B": 0.841, "Mixtral-8x7B": 0.856
            }
        },
    }
}


# ---------------------------------------------
# Helpers to build DataFrames on the fly
# ---------------------------------------------
def list_games():
    # games that exist in all settings (intersection)
    games_sets = [set(DATA[s].keys()) for s in DATA]
    return sorted(set.intersection(*games_sets))

def get_metrics(game):
    # metrics that exist in all settings for the chosen game (intersection)
    msets = []
    for setting in DATA:
        msets.append(set(DATA[setting][game].keys()))
    return sorted(set.intersection(*msets))

def df_metric_across_settings(game, metric):
    # rows=models, cols=settings
    frame = {}
    for setting in DATA:
        frame[setting] = DATA[setting][game][metric]
    df = pd.DataFrame(frame)
    # enforce model order
    return df.loc[MODELS]

# ---------------------------------------------
# Plotters
# ---------------------------------------------
def plot_metric_across_settings(game, metric, normalize=False):
    """
    Grouped BAR + LINE plots for a single metric in a game,
    comparing Base vs Prompt Injection vs SFT across models.
    """
    df = df_metric_across_settings(game, metric)
    plot_df = df.copy()
    if normalize:
        # min-max per column to show relative differences within each setting
        plot_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-12)

    # Bar chart
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"{game} — {metric} | Base vs Prompt Injection vs SFT" + (" (normalized)" if normalize else ""))
    ax.set_xlabel("Models")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Setting", loc="best")
    # annotate bars with the original (non-normalized) values
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=8, rotation=0)
    plt.tight_layout()
    plt.show()

    # Line chart
    ax2 = plot_df.plot(kind="line", marker="o", figsize=(10, 6))
    ax2.set_title(f"{game} — {metric} trends across settings" + (" (normalized)" if normalize else ""))
    ax2.set_xlabel("Models")
    ax2.set_ylabel(metric)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(title="Setting", loc="best")
    plt.tight_layout()
    plt.show()

def plot_delta_from_base(game, metric):
    """
    Bar chart of (Prompt Injection - Base) and (SFT - Base) for each model.
    Positive = increase vs Base, Negative = decrease vs Base.
    """
    df = df_metric_across_settings(game, metric)
    delta = pd.DataFrame({
        "Prompt Injection - Base": df["Prompt Injection"] - df["Base"],
        "SFT - Base": df["SFT"] - df["Base"]
    })
    ax = delta.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"Δ vs Base | {game} — {metric}")
    ax.set_xlabel("Models")
    ax.set_ylabel("Delta from Base")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.show()

def heatmap_game(setting, game, normalize=False):
    """
    Heatmap for a given setting and game (metrics × models).
    """
    metrics = get_metrics(game)
    df = pd.DataFrame({m: DATA[setting][game][m] for m in metrics}).T[MODELS]
    plot_df = df.copy()
    if normalize:
        # min-max per metric row
        plot_df = (plot_df.T - plot_df.min(axis=1)) / (plot_df.max(axis=1) - plot_df.min(axis=1) + 1e-12)
        plot_df = plot_df.T

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(metrics) + 2))
    im = ax.imshow(plot_df.values, aspect="auto")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_title(f"Heatmap — {setting} — {game}" + (" (normalized)" if normalize else ""))
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# QUICK USAGE EXAMPLES
# ---------------------------------------------
# See available games and metrics
# print(list_games())
# print(get_metrics("Prisoner's Dilemma"))

# 1) Grouped bar + line for a single metric
# plot_metric_across_settings("Prisoner's Dilemma", "Cooperation Freq.")
# plot_metric_across_settings("Social Context", "Utility", normalize=True)

# 2) Deltas vs Base (nice for seeing uplift/drop)
# plot_delta_from_base("Non-Atomic Congestion", "RHA")
# plot_delta_from_base("Dictator Game", "Theta")

# 3) Heatmap of a whole game within one setting
#heatmap_game("SFT", "Atomic Congestion")
#heatmap_game("Prompt Injection", "Social Context", normalize=True)
