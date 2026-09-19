# export_results.py – Export W&B results to tables and figures for thesis
# Usage: python export_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'


def load_results(csv_file="sweep_results.csv"):
    """Load sweep results from CSV file"""
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("Run sweep first to generate results")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} runs from {csv_file}")
    return df


def create_reward_ablation_table(df):
    """Create LaTeX table for reward ablation results"""
    
    # Filter for reward ablation runs
    reward_df = df[df['run_name'].str.contains('tb3-nav-reward-ablation', na=False)]
    if reward_df.empty:
        reward_df = df  # Use all data if not filtered
    
    # Group by reward parameters
    grouped = reward_df.groupby(['goal_reward', 'collision_penalty', 
                                  'progress_weight', 'away_penalty']).agg({
        'success_rate': ['mean', 'std'],
        'collision_rate': ['mean', 'std'],
        'avg_steps': ['mean', 'std']
    }).round(2)
    
    # Sort by success rate
    grouped = grouped.sort_values(('success_rate', 'mean'), ascending=False)
    
    # Generate LaTeX table
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Reward Function Ablation Study Results}" + "\n"
    latex += r"\label{tab:reward_ablation}" + "\n"
    latex += r"\begin{tabular}{lcccccc}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"Configuration & Success (\%) & Collision (\%) & Steps & Efficiency \\" + "\n"
    latex += r"(Goal/Col/Prog/Away) &  &  &  & \\" + "\n"
    latex += r"\hline" + "\n"
    
    for idx, row in grouped.head(15).iterrows():
        conf = f"{idx[0]}/{idx[1]}/{idx[2]:.1f}/{idx[3]:.1f}"
        success = f"{row[('success_rate','mean')]:.1f}$\pm${row[('success_rate','std')]:.1f}"
        collision = f"{row[('collision_rate','mean')]:.1f}$\pm${row[('collision_rate','std')]:.1f}"
        steps = f"{row[('avg_steps','mean')]:.0f}$\pm${row[('avg_steps','std')]:.0f}"
        
        latex += f"{conf} & {success} & {collision} & {steps} &  \\\\" + "\n"
    
    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"
    
    # Save to file
    with open('reward_ablation_table.tex', 'w') as f:
        f.write(latex)
    print("✅ Saved reward_ablation_table.tex")
    
    return latex


def create_hyperparam_table(df):
    """Create LaTeX table for hyperparameter sensitivity"""
    
    # Filter for hyperparameter runs
    hyper_df = df[df['run_name'].str.contains('hyperparam', na=False)]
    if hyper_df.empty:
        hyper_df = df
    
    # Group by learning rate and batch size
    grouped = hyper_df.groupby(['learning_rate', 'batch_size', 'gamma']).agg({
        'success_rate': ['mean', 'std'],
    }).round(2)
    
    grouped = grouped.sort_values(('success_rate', 'mean'), ascending=False)
    
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Hyperparameter Sensitivity Analysis}" + "\n"
    latex += r"\label{tab:hyperparam_sensitivity}" + "\n"
    latex += r"\begin{tabular}{lcccc}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"Learning Rate & Batch Size & Gamma & Success (\%) \\" + "\n"
    latex += r"\hline" + "\n"
    
    for idx, row in grouped.head(10).iterrows():
        lr = f"{idx[0]:.1e}" if isinstance(idx[0], float) else str(idx[0])
        bs = idx[1]
        gamma = f"{idx[2]:.3f}"
        success = f"{row[('success_rate','mean')]:.1f}$\pm${row[('success_rate','std')]:.1f}"
        
        latex += f"{lr} & {bs} & {gamma} & {success} \\\\" + "\n"
    
    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"
    
    with open('hyperparam_table.tex', 'w') as f:
        f.write(latex)
    print("✅ Saved hyperparam_table.tex")
    
    return latex


def create_architecture_comparison(df):
    """Create LaTeX table and bar chart for architecture comparison"""
    
    # Filter for architecture comparison runs
    arch_df = df[df['run_name'].str.contains('architecture', na=False)]
    if arch_df.empty:
        arch_df = df
    
    # Group by algorithm and hybrid flag
    grouped = arch_df.groupby(['algorithm', 'hybrid']).agg({
        'success_rate': ['mean', 'std'],
        'collision_rate': ['mean', 'std'],
        'avg_steps': ['mean', 'std']
    }).round(2)
    
    # Generate LaTeX table
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Architecture Comparison Results}" + "\n"
    latex += r"\label{tab:architecture_comparison}" + "\n"
    latex += r"\begin{tabular}{lcccccc}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"Algorithm & Hybrid & Success (\%) & Collision (\%) & Steps & Efficiency \\" + "\n"
    latex += r"\hline" + "\n"
    
    for idx, row in grouped.iterrows():
        alg = idx[0]
        hybrid = "Yes" if idx[1] else "No"
        success = f"{row[('success_rate','mean')]:.1f}$\pm${row[('success_rate','std')]:.1f}"
        collision = f"{row[('collision_rate','mean')]:.1f}$\pm${row[('collision_rate','std')]:.1f}"
        
        latex += f"{alg} & {hybrid} & {success} & {collision} & - & - \\\\" + "\n"
    
    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"
    
    with open('architecture_table.tex', 'w') as f:
        f.write(latex)
    print("✅ Saved architecture_table.tex")
    
    # Create bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate bar chart
    success_data = arch_df.groupby(['algorithm', 'hybrid'])['success_rate'].mean().unstack()
    success_data.plot(kind='bar', ax=axes[0], legend=True)
    axes[0].set_xlabel('Algorithm')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate by Architecture')
    axes[0].set_ylim(0, 100)
    axes[0].legend(title='Hybrid', labels=['No', 'Yes'])
    axes[0].grid(True, alpha=0.3)
    
    # Collision rate bar chart
    collision_data = arch_df.groupby(['algorithm', 'hybrid'])['collision_rate'].mean().unstack()
    collision_data.plot(kind='bar', ax=axes[1], legend=True)
    axes[1].set_xlabel('Algorithm')
    axes[1].set_ylabel('Collision Rate (%)')
    axes[1].set_title('Collision Rate by Architecture')
    axes[1].set_ylim(0, 100)
    axes[1].legend(title='Hybrid', labels=['No', 'Yes'])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved architecture_comparison.png")
    
    return fig


def create_learning_curves_plot():
    """Create placeholder for learning curves (would need data from W&B)"""
    print("\n⚠️ Learning curves require data from individual runs.")
    print("   To generate learning curves, use the W&B API to download step-by-step metrics.")
    print("   Example: run.history(keys=['eval/success_rate', 'episode_reward_mean'])")


def create_heatmap(df):
    """Create heatmap of success rate vs reward parameters"""
    if 'goal_reward' not in df.columns or 'progress_weight' not in df.columns:
        print("Cannot create heatmap: missing required columns")
        return
    
    # Pivot table for heatmap
    pivot = df.pivot_table(
        values='success_rate', 
        index='goal_reward', 
        columns='progress_weight',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Progress Weight')
    ax.set_ylabel('Goal Reward')
    ax.set_title('Success Rate Heatmap: Goal Reward vs Progress Weight')
    
    plt.colorbar(im, label='Success Rate (%)')
    plt.tight_layout()
    plt.savefig('reward_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved reward_heatmap.png")


def export_to_excel(df, filename="all_results.xlsx"):
    """Export all results to Excel for easy analysis"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # All results
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Summary by configuration
        if 'goal_reward' in df.columns:
            summary = df.groupby(['goal_reward', 'collision_penalty', 
                                  'progress_weight', 'away_penalty']).agg({
                'success_rate': ['mean', 'std', 'count'],
                'collision_rate': ['mean', 'std']
            }).round(2)
            summary.to_excel(writer, sheet_name='Reward Ablation Summary')
        
        # Summary by algorithm
        if 'algorithm' in df.columns:
            arch_summary = df.groupby(['algorithm', 'hybrid']).agg({
                'success_rate': ['mean', 'std', 'count'],
                'collision_rate': ['mean', 'std']
            }).round(2)
            arch_summary.to_excel(writer, sheet_name='Architecture Summary')
        
        # Summary by hyperparameters
        if 'learning_rate' in df.columns:
            hyper_summary = df.groupby(['learning_rate', 'batch_size']).agg({
                'success_rate': ['mean', 'std', 'count']
            }).round(2)
            hyper_summary.to_excel(writer, sheet_name='Hyperparameter Summary')
    
    print(f"✅ Exported all results to {filename}")


def main():
    print("="*70)
    print("W&B Results Export for Thesis")
    print("="*70 + "\n")
    
    # Load results
    df = load_results("sweep_results.csv")
    if df is None:
        print("\n📋 No results found. Run a sweep first using:")
        print("   python run_sweep.py")
        return
    
    print(f"\n📊 Data Overview:")
    print(f"   - Total runs: {len(df)}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Success rate range: {df['success_rate'].min():.1f}% - {df['success_rate'].max():.1f}%")
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Generate tables and figures
    print("\n" + "-"*50)
    print("Generating tables and figures...")
    print("-"*50)
    
    # 1. Reward ablation table
    if 'goal_reward' in df.columns:
        create_reward_ablation_table(df)
    
    # 2. Hyperparameter table
    if 'learning_rate' in df.columns:
        create_hyperparam_table(df)
    
    # 3. Architecture comparison
    if 'algorithm' in df.columns:
        create_architecture_comparison(df)
    
    # 4. Heatmap
    if 'goal_reward' in df.columns and 'progress_weight' in df.columns:
        create_heatmap(df)
    
    # 5. Export to Excel
    export_to_excel(df, "results/all_results.xlsx")
    
    print("\n" + "="*70)
    print("✅ All results exported!")
    print("="*70)
    print("\nFiles generated:")
    print("  - reward_ablation_table.tex  (LaTeX table)")
    print("  - hyperparam_table.tex        (LaTeX table)")
    print("  - architecture_table.tex      (LaTeX table)")
    print("  - architecture_comparison.png (bar chart)")
    print("  - reward_heatmap.png          (heatmap)")
    print("  - results/all_results.xlsx    (Excel workbook)")
    print("\nTo insert into Word:")
    print("  - For tables: Open .tex file, copy content to Word")
    print("  - For figures: Insert → Pictures → select PNG file")
    print("  - For Excel: Open .xlsx, copy-paste tables directly")


if __name__ == "__main__":
    main()
