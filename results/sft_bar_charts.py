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

# SFT (Supervised Fine-Tuning) Data
sft_data = {
    "Non-Atomic Congestion": {
        "RHA": [0.149, 0.342, 0.394, 0.391, 0.454, 0.580],
        "MIR": [0.822, 0.659, 0.427, 0.410, 0.451, 0.528],
        "DSR": [0.992, 0.982, 0.979, 0.981, 0.974, 0.974],
    },
    "Social Context": {
        "Deviation": [4.422, 2.578, 5.000, 5.000, 1.822, 5.000],
        "Rank": [0.010, 0.630, 0.000, 0.000, 0.785, 0.000],
    },
    "Dictator Game": {
        "Util. Gain": [1.239, 1.152, 0.441, -0.606, -0.074, 1.417],
        "Warm-Glow": [69.047, 49.570, 43.443, 46.404, 43.304, 45.637],
    },
    "Atomic Congestion": {
        "Social Welfare": [-5.567, -5.333, -192.503, -3.600, -6.667, -3.267],
        "SVO Angle": [-2.399, -2.314, 0.000, 0.000, -2.374, 0.000],
    },
    "Cost Sharing": {
        "NCC": [1.057, 1.057, 1.056, 1.057, 1.058, 1.053],
        "FS Index": [0.065, 0.065, 0.069, 0.067, 0.067, 0.063],
    },
    "Prisoner's Dilemma": {
        "Cooperation Freq.": [0.728, 0.344, 0.248, 0.228, 0.400, 0.876],
        "Avg. Payoff": [0.434, 0.657, 0.573, 0.447, 0.500, 0.370],
        "MCS": [0.862, 0.716, 0.569, 0.618, 0.670, 0.920],
    },
    "Hedonic Game": {
        "Altruism Score": [0.036, 0.155, 0.065, 0.024, 0.125, 0.125]
    },
    "Coalition Game": {
        "Altruism Score": [0.962, 0.948, 0.9158, 0.943, 0.841, 0.856]
    },
}

# Color palette for models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
model_colors = dict(zip(models, colors))

def plot_game_bar_chart(game_name: str, normalize: bool = False, save_fig: bool = True):
    """
    Create multiple bar charts in one figure for all metrics in the chosen game.
    
    Parameters
    ----------
    game_name : str
        Name of the game (must be a key in sft_data).
    normalize : bool, optional
        If True, scales each metric's values between 0 and 1
        to highlight relative differences (useful when ranges differ a lot).
    save_fig : bool, optional
        If True, saves the figure as PNG file.
    """
    if game_name not in sft_data:
        raise ValueError(f"Game '{game_name}' not found. Choose from: {list(sft_data.keys())}")
    
    game_data = sft_data[game_name]
    metrics = list(game_data.keys())
    num_metrics = len(metrics)
    
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
    fig.suptitle(f'{game_name} - Supervised Fine-Tuning (SFT) Results', fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = game_data[metric]
        
        if normalize:
            # Scale between 0 and 1 for visibility
            vals = np.array(values)
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        else:
            vals = values
        
        # Create bars with different colors for each model
        bars = ax.bar(range(len(models)), vals, 
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
        for i, (bar, actual) in enumerate(zip(bars, values)):
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
        filename = f"{game_name.replace(' ', '_').replace(chr(39), '').lower()}_sft.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()

def plot_all_games(normalize: bool = False, save_figs: bool = True):
    """
    Create bar charts for all games at once.
    
    Parameters
    ----------
    normalize : bool, optional
        If True, scales each metric's values between 0 and 1.
    save_figs : bool, optional
        If True, saves each figure as PNG file.
    """
    for game_name in sft_data.keys():
        print(f"\nCreating bar chart for: {game_name}")
        plot_game_bar_chart(game_name, normalize=normalize, save_fig=save_figs)

def create_summary_comparison():
    """
    Create a comprehensive comparison showing key metrics across all games.
    """
    # Select one key metric from each game for overview
    key_metrics = {
        "Non-Atomic Congestion": "RHA",
        "Social Context": "Rank", 
        "Dictator Game": "Warm-Glow",
        "Atomic Congestion": "Social Welfare",
        "Cost Sharing": "NCC",
        "Prisoner's Dilemma": "Cooperation Freq.",
        "Hedonic Game": "Altruism Score",
        "Coalition Game": "Altruism Score"
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Key Metrics Summary - Supervised Fine-Tuning (SFT) Results', fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (game, metric) in enumerate(key_metrics.items()):
        ax = axes[idx]
        values = sft_data[game][metric]
        
        bars = ax.bar(range(len(models)), values,
                     color=[model_colors[model] for model in models],
                     edgecolor='black', linewidth=0.8, alpha=0.8)
        
        ax.set_title(f'{game}\n{metric}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Models', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        
        # Annotate bars
        for bar, actual in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{actual:.3f}', ha='center',
                   va='bottom' if height >= 0 else 'top',
                   fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sft_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: sft_summary.png")
    plt.show()

def create_altruism_focused_chart():
    """
    Create a focused chart on altruism-related metrics.
    """
    altruism_metrics = {
        "Prisoner's Dilemma - Cooperation": sft_data["Prisoner's Dilemma"]["Cooperation Freq."],
        "Hedonic Game - Altruism": sft_data["Hedonic Game"]["Altruism Score"],
        "Coalition Game - Altruism": sft_data["Coalition Game"]["Altruism Score"],
        "Dictator Game - Util. Gain": sft_data["Dictator Game"]["Util. Gain"]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Altruism-Focused Metrics - Supervised Fine-Tuning (SFT) Results', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(altruism_metrics.items()):
        ax = axes[idx]
        
        bars = ax.bar(range(len(models)), values,
                     color=[model_colors[model] for model in models],
                     edgecolor='black', linewidth=0.8, alpha=0.8)
        
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Models', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        
        # Annotate bars
        for bar, actual in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{actual:.3f}', ha='center',
                   va='bottom' if height >= 0 else 'top',
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('altruism_focused_sft.png', dpi=300, bbox_inches='tight')
    print("Saved: altruism_focused_sft.png")
    plt.show()

def create_extreme_values_analysis():
    """
    Create a special analysis highlighting extreme or notable values in SFT results.
    """
    # Notable observations from the data
    extreme_cases = {
        "Atomic Congestion - Social Welfare": {
            "values": sft_data["Atomic Congestion"]["Social Welfare"],
            "note": "Gemini-2.5-F shows extreme negative value (-192.503)"
        },
        "Social Context - Rank": {
            "values": sft_data["Social Context"]["Rank"],
            "note": "Several models show 0.000 rank values"
        },
        "Prisoner's Dilemma - Cooperation": {
            "values": sft_data["Prisoner's Dilemma"]["Cooperation Freq."],
            "note": "Wide variation: Mixtral-8x7B highest (0.876), LLaMA-3.3 lowest (0.228)"
        },
        "Coalition Game - Altruism": {
            "values": sft_data["Coalition Game"]["Altruism Score"],
            "note": "Generally high altruism scores across all models"
        }
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Notable Patterns in SFT Results - Extreme Values Analysis', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (metric_name, data) in enumerate(extreme_cases.items()):
        ax = axes[idx]
        values = data["values"]
        
        bars = ax.bar(range(len(models)), values,
                     color=[model_colors[model] for model in models],
                     edgecolor='black', linewidth=0.8, alpha=0.8)
        
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Models', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        
        # Annotate bars
        for bar, actual in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{actual:.3f}', ha='center',
                   va='bottom' if height >= 0 else 'top',
                   fontsize=8, fontweight='bold')
        
        # Add note as subtitle
        ax.text(0.5, -0.15, data["note"], transform=ax.transAxes, ha='center', 
               fontsize=8, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig('sft_extreme_values_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: sft_extreme_values_analysis.png")
    plt.show()

if __name__ == "__main__":
    print("Creating Supervised Fine-Tuning (SFT) bar charts...")
    
    plot_game_bar_chart("Social Context")
    
    print("\nDone! Check the generated PNG files.")
