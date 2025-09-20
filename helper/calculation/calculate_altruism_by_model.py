#!/usr/bin/env python3
"""
Calculate normalized altruism score (A_bar) by LLM model for hedonic game results
Using the formula: A_bar = (1 / (T * |N|)) * sum_{t=1}^{T} sum_{i in N} Altruism_i^(t)
"""

import pandas as pd

def calculate_altruism_by_model(csv_file):
    """
    Calculate the normalized altruism score A_bar for each LLM model
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    print("=== Altruism Scores by LLM Model ===")
    print("=" * 80)
    
    # Get unique models
    models = df['llm_name'].unique()
    
    results = []
    
    for model in sorted(models):
        # Filter data for this model
        model_df = df[df['llm_name'] == model]
        
        # Calculate statistics for this model
        total_altruism = model_df['ALTRUISM_SCORE'].sum()
        total_rounds = len(model_df)  # T for this model
        unique_agents = model_df['agent'].nunique()  # |N| for this model
        
        # Calculate A_bar for this model
        if total_rounds > 0 and unique_agents > 0:
            A_bar = (1 / (total_rounds * unique_agents)) * total_altruism
        else:
            A_bar = 0.0
        
        # Additional statistics
        mean_altruism = model_df['ALTRUISM_SCORE'].mean()
        median_altruism = model_df['ALTRUISM_SCORE'].median()
        min_altruism = model_df['ALTRUISM_SCORE'].min()
        max_altruism = model_df['ALTRUISM_SCORE'].max()
        std_altruism = model_df['ALTRUISM_SCORE'].std()
        
        # Count non-zero scores
        non_zero_count = (model_df['ALTRUISM_SCORE'] > 0).sum()
        zero_count = (model_df['ALTRUISM_SCORE'] == 0).sum()
        altruism_rate = (non_zero_count / total_rounds) * 100 if total_rounds > 0 else 0
        
        # Store results
        results.append({
            'model': model,
            'A_bar': A_bar,
            'total_rounds': total_rounds,
            'unique_agents': unique_agents,
            'total_altruism': total_altruism,
            'mean_altruism': mean_altruism,
            'altruism_rate': altruism_rate,
            'max_altruism': max_altruism
        })
        
        # Print detailed results for this model
        print(f"Model: {model}")
        print(f"  A_bar (Normalized): {A_bar:.6f}")
        print(f"  Total rounds: {total_rounds}")
        print(f"  Unique agents: {unique_agents}")
        print(f"  Total altruism: {total_altruism:.2f}")
        print(f"  Mean altruism: {mean_altruism:.4f}")
        print(f"  Altruism rate: {altruism_rate:.1f}% ({non_zero_count}/{total_rounds})")
        print(f"  Min/Max: {min_altruism:.2f}/{max_altruism:.2f}")
        print(f"  Std dev: {std_altruism:.4f}")
        print()
    
    # Create summary DataFrame and sort by A_bar
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('A_bar', ascending=False)
    
    print("=" * 80)
    print("=== RANKING BY NORMALIZED ALTRUISM SCORE (A_bar) ===")
    print("=" * 80)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i}. {row['model']}")
        print(f"   A_bar: {row['A_bar']:.6f}")
        print(f"   Altruism rate: {row['altruism_rate']:.1f}%")
        print(f"   Mean altruism: {row['mean_altruism']:.4f}")
        print(f"   Max altruism: {row['max_altruism']:.2f}")
        print()
    
    # Calculate overall statistics
    print("=" * 80)
    print("=== OVERALL STATISTICS ===")
    print("=" * 80)
    
    overall_A_bar = df['ALTRUISM_SCORE'].sum() / (len(df) * df['agent'].nunique())
    overall_mean = df['ALTRUISM_SCORE'].mean()
    overall_altruism_rate = (df['ALTRUISM_SCORE'] > 0).sum() / len(df) * 100
    
    print(f"Overall A_bar: {overall_A_bar:.6f}")
    print(f"Overall mean altruism: {overall_mean:.4f}")
    print(f"Overall altruism rate: {overall_altruism_rate:.1f}%")
    print()
    
    # Model comparison
    print("=" * 80)
    print("=== MODEL COMPARISON ===")
    print("=" * 80)
    
    best_model = results_df.iloc[0]
    worst_model = results_df.iloc[-1]
    
    print(f"Most altruistic model: {best_model['model']}")
    print(f"  A_bar: {best_model['A_bar']:.6f}")
    print(f"  Altruism rate: {best_model['altruism_rate']:.1f}%")
    print()
    
    print(f"Least altruistic model: {worst_model['model']}")
    print(f"  A_bar: {worst_model['A_bar']:.6f}")
    print(f"  Altruism rate: {worst_model['altruism_rate']:.1f}%")
    print()
    
    # Calculate variance between models
    A_bar_values = results_df['A_bar'].values
    variance = ((A_bar_values - A_bar_values.mean()) ** 2).mean()
    std_dev = variance ** 0.5
    
    print(f"Variance in A_bar across models: {variance:.8f}")
    print(f"Standard deviation: {std_dev:.6f}")
    print()
    
    return results_df

def main():
    csv_file = "data/hedonic_altruistic_test_20250920_010551.csv"
    
    try:
        results_df = calculate_altruism_by_model(csv_file)
        
        # Save results to CSV
        output_file = "altruism_by_model_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
