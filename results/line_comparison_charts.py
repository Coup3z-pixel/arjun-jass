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

# Base Models Data
base_data = {
    "Non-Atomic Congestion": {
        "RHA": [0.423, 0.429, 0.427, 0.423, 0.502, 0.508],
        "MIR": [0.472, 0.474, 0.475, 0.472, 0.495, 0.510],
        "DSR": [-16.200, -16.510, -16.580, -14.730, 0.971, 0.971],
    },
    "Social Context": {
        "Deviation": [-6.503, -6.005, -6.160, -5.953, 1.478, 2.700],
        "Rank": [0.600, 0.594, 0.588, 0.580, 0.926, 0.714],
    },
    "Dictator Game": {
        "Util. Gain": [-2.000, 2.250, -0.125, 2.167, 0.191, -0.472],
        "Warm-Glow": [33.430, 43.860, 47.220, 67.150, 40.010, 38.680],
    },
    "Atomic Congestion": {
        "Social Welfare": [-34.400, -34.770, -34.630, -34.570, -6.470, -6.600],
        "SVO Angle": [-2.281, -2.387, -2.353, -2.324, -2.309, -2.352],
    },
    "Cost Sharing": {
        "NCC": [1.070, 1.072, 1.071, 1.071, 1.058, 1.058],
        "FS Index": [0.055, 0.056, 0.054, 0.052, 0.067, 0.067],
    },
    "Prisoner's Dilemma": {
        "Cooperation Freq.": [0.449, 0.461, 0.473, 0.475, 0.408, 0.412],
        "Avg. Payoff": [0.536, 0.449, 0.580, 0.553, 0.647, 0.500],
        "MCS": [0.539, 0.508, 0.464, 0.525, 0.625, 0.592],
    },
    "Hedonic Game": {
        "Altruism Score": [0.095, 0.085, 0.125, 0.125, 0.054, 0.030]
    },
    "Coalition Game": {
        "Altruism Score": [0.642, 0.812, 0.623, 0.923, 0.772, 0.783]
    },
}

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

# SFT Data (updated with correct Gemini-2.5-F data)
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

def plot_line_comparison(game_name: str, save_fig: bool = True):
    """
    Create line charts comparing Base, Prompt Injection, and SFT for each metric.
    
    Parameters
    ----------
    game_name : str
        Name of the game to plot.
    save_fig : bool, optional
        If True, saves the figure as PNG file.
    """
    if game_name not in base_data:
        raise ValueError(f"Game '{game_name}' not found. Choose from: {list(base_data.keys())}")
    
    metrics = list(base_data[game_name].keys())
    num_metrics = len(metrics)
    
    # Determine subplot layout
    if num_metrics == 1:
        rows, cols = 1, 1
        figsize = (12, 8)
    elif num_metrics == 2:
        rows, cols = 1, 2
        figsize = (20, 8)
    elif num_metrics == 3:
        rows, cols = 1, 3
        figsize = (24, 8)
    else:
        rows = (num_metrics + 2) // 3
        cols = 3
        figsize = (24, 8 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'{game_name} - Base vs Prompt Injection vs SFT Comparison (Line Chart)', 
                 fontsize=18, fontweight='bold')
    
    # Handle single subplot case
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # X-axis positions
    x = np.arange(len(models))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        base_values = base_data[game_name][metric]
        prompt_values = prompt_data[game_name][metric]
        sft_values = sft_data[game_name][metric]
        
        # Create line plots for each condition
        ax.plot(x, base_values, 'o-', linewidth=3, markersize=8, 
               label='Base', color='#1f77b4', alpha=0.8)
        ax.plot(x, prompt_values, 's--', linewidth=3, markersize=8, 
               label='Prompt Injection', color='#ff7f0e', alpha=0.8)
        ax.plot(x, sft_values, '^:', linewidth=3, markersize=8,
               label='SFT', color='#2ca02c', alpha=0.8)
        
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (base_val, prompt_val, sft_val) in enumerate(zip(base_values, prompt_values, sft_values)):
            # Only annotate every other point to avoid crowding
            if i % 2 == 0:
                ax.annotate(f'{base_val:.3f}', (x[i], base_val), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                ax.annotate(f'{prompt_val:.3f}', (x[i], prompt_val), 
                           textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
                ax.annotate(f'{sft_val:.3f}', (x[i], sft_val), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Hide empty subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_fig:
        filename = f"{game_name.replace(' ', '_').replace('\'', '').lower()}_line_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()

def create_all_line_comparisons():
    """Create line comparison charts for all games."""
    for game_name in base_data.keys():
        print(f"\nCreating line comparison chart for: {game_name}")
        plot_line_comparison(game_name)

def create_key_metrics_line_comparison():
    """Create a focused line comparison of key altruism metrics across all conditions."""
    key_metrics = [
        ("Prisoner's Dilemma", "Cooperation Freq."),
        ("Coalition Game", "Altruism Score"),
        ("Hedonic Game", "Altruism Score"),
        ("Dictator Game", "Util. Gain"),
        ("Atomic Congestion", "Social Welfare"),
        ("Social Context", "Rank")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Key Altruism Metrics: Base vs Prompt Injection vs SFT (Line Charts)', 
                 fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    x = np.arange(len(models))
    
    for idx, (game, metric) in enumerate(key_metrics):
        ax = axes[idx]
        
        base_values = base_data[game][metric]
        prompt_values = prompt_data[game][metric]
        sft_values = sft_data[game][metric]
        
        # Create line plots
        ax.plot(x, base_values, 'o-', linewidth=3, markersize=10,
               label='Base', color='#1f77b4', alpha=0.8)
        ax.plot(x, prompt_values, 's--', linewidth=3, markersize=10,
               label='Prompt Injection', color='#ff7f0e', alpha=0.8)
        ax.plot(x, sft_values, '^:', linewidth=3, markersize=10,
               label='SFT', color='#2ca02c', alpha=0.8)
        
        ax.set_title(f'{game}\n{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Models', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add some key value annotations (first and last model)
        for i in [0, len(models)-1]:
            ax.annotate(f'{base_values[i]:.3f}', (x[i], base_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)
            ax.annotate(f'{prompt_values[i]:.3f}', (x[i], prompt_values[i]), 
                       textcoords="offset points", xytext=(0,-15), ha='center', fontsize=7)
            ax.annotate(f'{sft_values[i]:.3f}', (x[i], sft_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('key_altruism_metrics_line_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: key_altruism_metrics_line_comparison.png")
    plt.show()

def create_comprehensive_overview():
    """Create a comprehensive overview showing all games in one large figure."""
    fig, axes = plt.subplots(4, 2, figsize=(24, 20))
    fig.suptitle('Comprehensive Game Comparison: Base vs Prompt Injection vs SFT', 
                 fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    x = np.arange(len(models))
    
    # Select one key metric from each game
    game_metrics = [
        ("Non-Atomic Congestion", "RHA"),
        ("Social Context", "Rank"),
        ("Dictator Game", "Util. Gain"),
        ("Atomic Congestion", "Social Welfare"),
        ("Cost Sharing", "NCC"),
        ("Prisoner's Dilemma", "Cooperation Freq."),
        ("Hedonic Game", "Altruism Score"),
        ("Coalition Game", "Altruism Score")
    ]
    
    for idx, (game, metric) in enumerate(game_metrics):
        ax = axes[idx]
        
        base_values = base_data[game][metric]
        prompt_values = prompt_data[game][metric]
        sft_values = sft_data[game][metric]
        
        # Create line plots with different styles
        ax.plot(x, base_values, 'o-', linewidth=3, markersize=8,
               label='Base', color='#1f77b4', alpha=0.8)
        ax.plot(x, prompt_values, 's--', linewidth=3, markersize=8,
               label='Prompt Injection', color='#ff7f0e', alpha=0.8)
        ax.plot(x, sft_values, '^:', linewidth=3, markersize=8,
               label='SFT', color='#2ca02c', alpha=0.8)
        
        ax.set_title(f'{game} - {metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Models', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_game_comparison_lines.png', dpi=300, bbox_inches='tight')
    print("Saved: comprehensive_game_comparison_lines.png")
    plt.show()

if __name__ == "__main__":
    print("Creating line comparison charts...")
    
    # Create individual game line comparisons
    plot_line_comparison("Social Context")
    #plot_line_comparison("Dictator Game")
    #plot_line_comparison("Non-Atomic Congestion")
    
    # Create key metrics summary
    #print("\nCreating key metrics line comparison...")
    #create_key_metrics_line_comparison()
    
    # Create comprehensive overview
    #print("\nCreating comprehensive overview...")
    #create_comprehensive_overview()
    
    # Uncomment to create all games at once
    #create_all_line_comparisons()
    
    print("\nDone! Check the generated PNG files.")
