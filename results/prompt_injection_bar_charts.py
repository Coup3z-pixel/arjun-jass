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

# Prompt Injection Data
prompt_data = {
    "Non-Atomic Congestion": {
        "RHA": [0.289, 0.296, 0.291, 0.302, 0.400, 0.400],
        "MIR": [0.333, 0.344, 0.330, 0.342, 0.410, 0.411],
        "DSR": [-12.090, -11.650, -11.330, -11.780, 0.979, 0.978],
    },
    "Social Context": {
        "Deviation": [-5.664, -5.300, -4.948, -5.374, 3.552, 3.448],
        "Rank": [0.455, 0.452, 0.430, 0.469, 0.098, 0.094],
    },
    "Dictator Game": {
        "Util. Gain": [1.750, 0.333, 2.000, 2.583, 0.870, 1.305],
        "Warm-Glow": [59.430, 53.120, 54.630, 69.640, 50.625, 50.494],
    },
    "Atomic Congestion": {
        "Social Welfare": [-34.930, -34.730, -34.800, -34.670, -5.267, -5.233],
        "SVO Angle": [-2.411, -2.376, -2.395, -2.347, -2.393, -2.320],
    },
    "Cost Sharing": {
        "NCC": [1.061, 1.061, 1.061, 1.061, 1.058, 1.058],
        "FS Index": [0.063, 0.063, 0.063, 0.063, 0.066, 0.066],
    },
    "Prisoner's Dilemma": {
        "Cooperation Freq.": [0.996, 0.994, 0.984, 0.992, 0.984, 0.990],
        "Avg. Payoff": [0.586, 0.255, 0.479, 0.491, 0.677, 0.527],
        "MCS": [0.996, 1.000, 0.980, 0.988, 0.996, 0.996],
    },
    "Hedonic Game": {
        "Altruism Score": [0.125, 0.140, 0.125, 0.125, 0.109, 0.109]
    },
    "Coalition Game": {
        "Altruism Score": [0.861, 0.858, 0.858, 0.861, 0.875, 0.848]
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
        Name of the game (must be a key in prompt_data).
    normalize : bool, optional
        If True, scales each metric's values between 0 and 1
        to highlight relative differences (useful when ranges differ a lot).
    save_fig : bool, optional
        If True, saves the figure as PNG file.
    """
    if game_name not in prompt_data:
        raise ValueError(f"Game '{game_name}' not found. Choose from: {list(prompt_data.keys())}")
    
    game_data = prompt_data[game_name]
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
    fig.suptitle(f'{game_name} - Prompt Injection Results', fontsize=16, fontweight='bold')
    
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
        filename = f"{game_name.replace(' ', '_').replace(chr(39), '').lower()}_prompt_injection.png"
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
    for game_name in prompt_data.keys():
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
    fig.suptitle('Key Metrics Summary - Prompt Injection Results', fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (game, metric) in enumerate(key_metrics.items()):
        ax = axes[idx]
        values = prompt_data[game][metric]
        
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
    plt.savefig('prompt_injection_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: prompt_injection_summary.png")
    plt.show()

def create_altruism_focused_chart():
    """
    Create a focused chart on altruism-related metrics.
    """
    altruism_metrics = {
        "Prisoner's Dilemma - Cooperation": prompt_data["Prisoner's Dilemma"]["Cooperation Freq."],
        "Hedonic Game - Altruism": prompt_data["Hedonic Game"]["Altruism Score"],
        "Coalition Game - Altruism": prompt_data["Coalition Game"]["Altruism Score"],
        "Dictator Game - Util. Gain": prompt_data["Dictator Game"]["Util. Gain"]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Altruism-Focused Metrics - Prompt Injection Results', fontsize=16, fontweight='bold')
    
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
    plt.savefig('altruism_focused_prompt_injection.png', dpi=300, bbox_inches='tight')
    print("Saved: altruism_focused_prompt_injection.png")
    plt.show()

if __name__ == "__main__":
    print("Creating Prompt Injection bar charts...")
    
    # Uncomment to create all games at once
    plot_game_bar_chart("Social Context")
    
    print("\nDone! Check the generated PNG files.")
