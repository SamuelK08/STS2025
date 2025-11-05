import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

results_df = pd.read_csv('liana_merged_results.csv')

resistance_signals = results_df[
    (results_df['specificity_rank_NR'] <= 0.1) &
    (results_df['specificity_rank_R'] > 0.3) &
    (results_df['target'].isin(['Exhausted CD8+ T-cells', 'CD8+ T-cells', 'T-cells']))
].copy()

druggable = ['HAVCR2', 'PDCD1', 'CD274', 'PDCD1LG2', 'CTLA4', 'LAG3', 
             'TIGIT', 'BTLA', 'CD47', 'SIRPA', 'TNFRSF9', 'TNFRSF4', 'SIRPG']

druggable_hits = resistance_signals[
    resistance_signals['receptor_complex'].isin(druggable) |
    resistance_signals['ligand_complex'].isin(druggable)
].copy()

druggable_hits['priority_score'] = (
    (1 - druggable_hits['specificity_rank_NR']) * 0.4 +  
    (druggable_hits['lr_means_NR'] / druggable_hits['lr_means_NR'].max()) * 0.3 + 
    (druggable_hits['rank_diff']) * 0.3  
)

druggable_hits = druggable_hits.sort_values('priority_score', ascending=False)

print("=== TOP 10 DRUGGABLE TARGETS BY PRIORITY SCORE ===\n")
priority_cols = ['source', 'target', 'ligand_complex', 'receptor_complex', 
                'lr_means_NR', 'lr_means_R', 'specificity_rank_NR', 
                'rank_diff', 'priority_score']
print(druggable_hits[priority_cols].head(10).to_string())

print("\n\n=== CHECKPOINT INTERACTIONS BY SOURCE ===")
checkpoint_summary = druggable_hits.groupby(['receptor_complex', 'source']).agg({
    'lr_means_NR': 'mean',
    'specificity_rank_NR': 'mean',
    'ligand_complex': 'count'
}).round(3)
checkpoint_summary.columns = ['Avg_Expression', 'Avg_Specificity', 'Count']
checkpoint_summary = checkpoint_summary.sort_values(['receptor_complex', 'Count'], ascending=[True, False])
print(checkpoint_summary)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
top_20 = druggable_hits.head(20).copy()
top_20['interaction'] = top_20['source'].str[:15] + '\n→ ' + top_20['ligand_complex'] + '-' + top_20['receptor_complex']

heatmap_data = top_20[['interaction', 'specificity_rank_NR', 'lr_means_NR', 'rank_diff']].set_index('interaction')
heatmap_data.columns = ['Specificity\n(NR)', 'Expression\n(NR)', 'Rank\nDifference']
heatmap_data['Specificity\n(NR)'] = 1 - heatmap_data['Specificity\n(NR)']
heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

sns.heatmap(heatmap_data, cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'Normalized Score'})
ax1.set_title('Top 20 Druggable Targets - Quality Metrics', fontsize=12, weight='bold')
ax1.set_ylabel('')

ax2 = axes[0, 1]
target_counts = druggable_hits['receptor_complex'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(target_counts)))
target_counts.plot(kind='bar', ax=ax2, color=colors)
ax2.set_ylabel('Number of Interactions', fontsize=11)
ax2.set_xlabel('Checkpoint Receptor', fontsize=11)
ax2.set_title('Druggable Checkpoint Frequency', fontsize=12, weight='bold')
ax2.tick_params(axis='x', rotation=45)

ax3 = axes[1, 0]
source_checkpoint = druggable_hits.groupby(['source', 'receptor_complex']).size().unstack(fill_value=0)
source_checkpoint.plot(kind='barh', stacked=True, ax=ax3, colormap='tab20')
ax3.set_xlabel('Number of Interactions', fontsize=11)
ax3.set_ylabel('Source Cell Type', fontsize=11)
ax3.set_title('Checkpoint Signaling by Source Cell', fontsize=12, weight='bold')
ax3.legend(title='Receptor', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

ax4 = axes[1, 1]
for receptor in druggable_hits['receptor_complex'].unique():
    subset = druggable_hits[druggable_hits['receptor_complex'] == receptor]
    ax4.scatter(subset['specificity_rank_NR'], subset['lr_means_NR'], 
               label=receptor, s=100, alpha=0.7)

ax4.set_xlabel('Specificity Rank in NR (lower = better)', fontsize=11)
ax4.set_ylabel('Mean LR Expression in NR', fontsize=11)
ax4.set_title('Druggable Targets: Specificity vs Expression', fontsize=12, weight='bold')
ax4.axvline(x=0.05, color='red', linestyle='--', alpha=0.3, label='Top 5% cutoff')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.invert_xaxis()

plt.tight_layout()
plt.savefig('figures/druggable_targets_analysis.png', dpi=300, bbox_inches='tight')
print("\n\nSaved druggable targets analysis plot")
plt.show()

print("\n\n=== CLINICAL TRANSLATION PRIORITIES ===")
print("\n TIER 1: Ready for Clinical Investigation")
tier1 = druggable_hits[
    (druggable_hits['specificity_rank_NR'] <= 0.05) &
    (druggable_hits['lr_means_NR'] > 1.0)
]
print(f"Found {len(tier1)} high-priority targets:\n")
for idx, row in tier1.iterrows():
    print(f"  • {row['ligand_complex']} → {row['receptor_complex']}")
    print(f"    Source: {row['source']}")
    print(f"    Expression: {row['lr_means_NR']:.2f} (NR) vs {row['lr_means_R']:.2f} (R)")
    print(f"    Specificity rank: {row['specificity_rank_NR']:.4f}\n")

print("\n🔬 TIER 2: Novel Biology - Needs Validation")
tier2 = druggable_hits[
    (druggable_hits['specificity_rank_NR'] > 0.05) &
    (druggable_hits['specificity_rank_NR'] <= 0.1) &
    (druggable_hits['lr_means_NR'] > 0.8)
]
print(f"Found {len(tier2)} promising novel targets")
print(tier2[['source', 'ligand_complex', 'receptor_complex', 'lr_means_NR', 'specificity_rank_NR']].to_string())
