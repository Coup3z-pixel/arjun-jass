import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        "Beta": {
            "ChatGPT-4o": -2.000, "GPT-3.5-T": 2.250, "Gemini-2.5-F": -0.125,
            "LLaMA-3.3": 2.167, "Qwen3-14B": 0.191, "Mixtral-8x7B": -0.472
        },
        "Theta": {
            "ChatGPT-4o": 33.430, "GPT-3.5-T": 43.860, "Gemini-2.5-F": 47.220,
            "LLaMA-3.3": 67.150, "Qwen3-14B": 40.010, "Mixtral-8x7B": 38.680
        },
    },
    "Atomic Congestion": {
        "Social Welfare": {
            "ChatGPT-4o": -34.400, "GPT-3.5-T": -34.770, "Gemini-2.5-F": -34.630,
            "LLaMA-3.3": -34.570, "Qwen3-14B": -6.470, "Mixtral-8x7B": -6.600
        },
        "Inequity Aversion": {
            "ChatGPT-4o": -8.530, "GPT-3.5-T": -8.910, "Gemini-2.5-F": -8.570,
            "LLaMA-3.3": -8.790, "Qwen3-14B": -3.500, "Mixtral-8x7B": -3.870
        },
        "SVO Angle": {
            "ChatGPT-4o": -2.281, "GPT-3.5-T": -2.387, "Gemini-2.5-F": -2.353,
            "LLaMA-3.3": -2.324, "Qwen3-14B": -2.309, "Mixtral-8x7B": -2.352
        },
    },
    "Cost Sharing": {
        "Eq. 13": {
            "ChatGPT-4o": 1.070, "GPT-3.5-T": 1.072, "Gemini-2.5-F": 1.071,
            "LLaMA-3.3": 1.071, "Qwen3-14B": 1.058, "Mixtral-8x7B": 1.058
        },
        "Eq. 14": {
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

# Color palette for models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
model_colors = dict(zip(models, colors))

def plot_game_metrics(game_name: str, normalize: bool = False, save_fig: bool = True):
    """
    Create multiple bar plots in one figure for all metrics in the chosen game.
    
    Parameters
    ----------
    game_name : str
        Name of the game (must be a key in dfs).
    normalize : bool, optional
        If True, scales each metric's values between 0 and 1
        to highlight relative differences (useful when ranges differ a lot).
    save_fig : bool, optional
        If True, saves the figure as PNG file.
    """
    if game_name not in dfs:
        raise ValueError(f"Game '{game_name}' not found. Choose from: {list(dfs.keys())}")
    
    df = dfs[game_name]
    num_metrics = len(df.index)
    
    # Determine subplot layout
    if num_metrics == 1:
        rows, cols = 1, 1
        figsize = (10, 6)
    elif num_metrics == 2:
        rows, cols = 1, 2
        figsize = (15, 6)
    elif num_metrics == 3:
        rows, cols = 1, 3
        figsize = (18, 6)
    else:
        rows = (num_metrics + 2) // 3  # Ceiling division
        cols = 3
        figsize = (18, 6 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'{game_name} - All Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(df.index):
        ax = axes[idx]
        values = df.loc[metric]
        
        if normalize:
            # Scale between 0 and 1 for visibility
            vals = (values - values.min()) / (values.max() - values.min() + 1e-9)
        else:
            vals = values
        
        # Create bars with different colors for each model
        bars = ax.bar(range(len(models)), vals.values, 
                     color=[model_colors[model] for model in models],
                     edgecolor='black', linewidth=0.8, alpha=0.8)
        
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Models', fontsize=10)
        ax.set_ylabel(metric + (" (normalized)" if normalize else ""), fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        
        # Annotate each bar with actual value
        for i, (bar, actual) in enumerate(zip(bars, values.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{actual:.3f}', ha='center', 
                   va='bottom' if height >= 0 else 'top', 
                   fontsize=8, fontweight='bold')
    
    # Hide empty subplots if any
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_fig:
        filename = f"{game_name.replace(' ', '_').replace("'", "").lower()}_metrics.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()

def plot_all_games(normalize: bool = False, save_figs: bool = True):
    """
    Create charts for all games at once.
    
    Parameters
    ----------
    normalize : bool, optional
        If True, scales each metric's values between 0 and 1.
    save_figs : bool, optional
        If True, saves each figure as PNG file.
    """
    for game_name in data.keys():
        print(f"\nCreating chart for: {game_name}")
        plot_game_metrics(game_name, normalize=normalize, save_fig=save_figs)

if __name__ == "__main__":
    # Example usage:
    print("Creating individual game charts...")
    
    # Create charts for specific games
    plot_game_metrics("Social Context")
    
    