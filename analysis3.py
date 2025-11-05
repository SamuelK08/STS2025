import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_csv('liana_merged_results.csv')

resistance_signals = results_df[
    (results_df['specificity_rank_NR'] <= 0.1) &
    (results_df['specificity_rank_R'] > 0.3) &
    (results_df['target'].isin(['Exhausted CD8+ T-cells', 'CD8+ T-cells', 'T-cells']))
].copy()

source_counts = resistance_signals['source'].value_counts().head(10)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
source_counts.plot(kind='barh', ax=ax1, color='#d95f02')
ax1.set_xlabel('Number of Resistance Interactions')
ax1.set_title('Which Cell Types Send Suppressive Signals?')
ax1.invert_yaxis()

ax2 = axes[0, 1]
target_counts = resistance_signals['target'].value_counts()
target_counts.plot(kind='bar', ax=ax2, color='#7570b3')
ax2.set_ylabel('Number of Interactions')
ax2.set_title('Which T-cell Types Are Targeted?')
ax2.tick_params(axis='x', rotation=45)

ax3 = axes[1, 0]
ax3.scatter(resistance_signals['lr_means_NR'], resistance_signals['lr_means_R'], 
            alpha=0.5, s=30)
ax3.plot([0, 4], [0, 4], 'r--', alpha=0.5, label='Equal expression')
ax3.set_xlabel('Expression in Non-Responders')
ax3.set_ylabel('Expression in Responders')
ax3.set_title('Ligand-Receptor Expression Comparison')
ax3.legend()

ax4 = axes[1, 1]
ligand_counts = resistance_signals['ligand_complex'].value_counts().head(15)
ligand_counts.plot(kind='barh', ax=ax4, color='#1b9e77')
ax4.set_xlabel('Frequency')
ax4.set_title('Most Common Resistance Ligands')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/resistance_signal_overview.png', dpi=300)
print("Saved overview plot")
plt.show()

print("\n=== PATHWAY/FAMILY ANALYSIS ===")
resistance_signals['ligand_family'] = resistance_signals['ligand_complex'].apply(
    lambda x: 'MHC' if 'HLA-' in x or 'H2-' in x 
    else 'Integrin' if 'COL' in x or 'LAMA' in x or 'FN' in x or 'ICAM' in x or 'VCAM' in x
    else 'Checkpoint' if x in ['LGALS9', 'CD274', 'PDCD1LG2', 'BTLA', 'HAVCR2']
    else 'Chemokine' if 'CCL' in x or 'CXCL' in x
    else 'Other'
)

family_summary = resistance_signals.groupby('ligand_family').agg({
    'ligand_complex': 'count',
    'lr_means_NR': 'mean',
    'specificity_rank_NR': 'mean'
}).round(3)
family_summary.columns = ['Count', 'Avg_Expression_NR', 'Avg_Specificity_NR']
family_summary = family_summary.sort_values('Count', ascending=False)

print(family_summary)

print("\n=== DRUGGABLE TARGETS (Known or in trials) ===")
druggable = ['HAVCR2', 'PDCD1', 'CD274', 'PDCD1LG2', 'CTLA4', 'LAG3', 
             'TIGIT', 'BTLA', 'CD47', 'SIRPA', 'TNFRSF9', 'TNFRSF4']

druggable_hits = resistance_signals[
    resistance_signals['receptor_complex'].isin(druggable) |
    resistance_signals['ligand_complex'].isin(druggable)
]

print(f"\nFound {len(druggable_hits)} interactions involving known druggable targets:")
print(druggable_hits[['source', 'target', 'ligand_complex', 'receptor_complex', 
                      'lr_means_NR', 'specificity_rank_NR']].head(20).to_string())
