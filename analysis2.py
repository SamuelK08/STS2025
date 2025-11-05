import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

results_df = pd.read_csv('liana_merged_results.csv')

resistance_signals = results_df[
    (results_df['specificity_rank_NR'] <= 0.1) &
    (results_df['specificity_rank_R'] > 0.3) &
    (results_df['target'].isin(['Exhausted CD8+ T-cells', 'CD8+ T-cells', 'T-cells']))
].copy()

resistance_signals = resistance_signals.sort_values('rank_diff', ascending=False)

print("=== TOP 20 RESISTANCE SIGNALS WITH EXPRESSION DATA ===\n")
display_cols = ['source', 'target', 'ligand_complex', 'receptor_complex', 
                'lr_means_NR', 'lr_means_R', 'expr_prod_NR', 'expr_prod_R',
                'specificity_rank_NR', 'specificity_rank_R', 'magnitude_rank_NR', 'magnitude_rank_R']
print(resistance_signals[display_cols].head(20).to_string())

print("\n\n=== HIGH MAGNITUDE + HIGH SPECIFICITY SIGNALS ===")
high_quality = results_df[
    (results_df['specificity_rank_NR'] <= 0.05) &
    (results_df['magnitude_rank_NR'] <= 0.1) & 
    (results_df['specificity_rank_R'] > 0.3) &
    (results_df['target'].isin(['Exhausted CD8+ T-cells', 'CD8+ T-cells', 'T-cells']))
].copy()

high_quality = high_quality.sort_values('rank_diff', ascending=False)
print(high_quality[display_cols].head(20).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
scatter1 = ax1.scatter(resistance_signals['magnitude_rank_NR'], 
                       resistance_signals['specificity_rank_NR'],
                       c=resistance_signals['lr_means_NR'],
                       cmap='Reds', s=50, alpha=0.6)
ax1.set_xlabel('Magnitude Rank (Expression Level)')
ax1.set_ylabel('Specificity Rank')
ax1.set_title('Non-Responders: T-cell Targeting Interactions')
ax1.invert_xaxis()
ax1.invert_yaxis()
plt.colorbar(scatter1, ax=ax1, label='Mean LR Expression')

ax2 = axes[1]
scatter2 = ax2.scatter(resistance_signals['magnitude_rank_R'], 
                       resistance_signals['specificity_rank_R'],
                       c=resistance_signals['lr_means_R'],
                       cmap='Blues', s=50, alpha=0.6)
ax2.set_xlabel('Magnitude Rank (Expression Level)')
ax2.set_ylabel('Specificity Rank')
ax2.set_title('Responders: T-cell Targeting Interactions')
ax2.invert_xaxis()
ax2.invert_yaxis()
plt.colorbar(scatter2, ax=ax2, label='Mean LR Expression')

plt.tight_layout()
plt.savefig('figures/specificity_vs_magnitude.png', dpi=300)
print("\n\nSaved specificity vs magnitude plot")
plt.show()

print("\n=== SUMMARY STATISTICS ===")
print(f"\nTotal resistance signals identified: {len(resistance_signals)}")
print(f"\nHigh-quality signals (top 5% specificity, top 10% magnitude): {len(high_quality)}")

print("\n--- Expression levels comparison ---")
print(f"Mean LR expression in NR: {resistance_signals['lr_means_NR'].mean():.4f}")
print(f"Mean LR expression in R: {resistance_signals['lr_means_R'].mean():.4f}")
print(f"Fold difference: {resistance_signals['lr_means_NR'].mean() / resistance_signals['lr_means_R'].mean():.2f}x")

print("\n=== KEY INTERACTIONS FOR EXPERIMENTAL VALIDATION ===")
key_interactions = high_quality.head(10)
for idx, row in key_interactions.iterrows():
    print(f"\n{row['ligand_complex']} → {row['receptor_complex']}")
    print(f"  Source: {row['source']} → Target: {row['target']}")
    print(f"  Expression NR: {row['lr_means_NR']:.3f} | R: {row['lr_means_R']:.3f}")
    print(f"  Specificity NR: {row['specificity_rank_NR']:.6f} | R: {row['specificity_rank_R']:.6f}")
