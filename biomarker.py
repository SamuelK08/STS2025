import pandas as pd
import numpy as np
import scanpy as sc
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

adata = sc.read('processed_melanoma_data_fortified.h5ad')
pre_mask = adata.obs['Time'].str.contains('Pre', na=False)
adata_pre = adata[pre_mask].copy()

print("=== PATIENT-LEVEL BIOMARKER ANALYSIS ===\n")
print(f"Total pre-treatment cells: {adata_pre.n_obs}")
print(f"Total patients: {adata_pre.obs['patient_ID'].nunique()}")
print(f"Responders: {(adata_pre.obs['Response'] == 'Responder').sum()} cells")
print(f"Non-responders: {(adata_pre.obs['Response'] == 'Non-responder').sum()} cells")

def calculate_patient_score(adata, patient_col='patient_ID'):
    """Calculate LGALS9-TIM3 score per patient"""
    
    lgals9_idx = adata.var_names.get_loc('LGALS9')
    havcr2_idx = adata.var_names.get_loc('HAVCR2')
    
    patient_scores = []
    
    for patient in adata.obs[patient_col].unique():
        patient_mask = adata.obs[patient_col] == patient
        
        source_cells = patient_mask & adata.obs['cell_type'].isin([
            'Regulatory T-cells (Tregs)', 'Malignant', 'Endothelial'
        ])
        
        target_cells = patient_mask & (adata.obs['cell_type'] == 'Exhausted CD8+ T-cells')
        
        if source_cells.sum() > 0:
            lgals9_expr = adata.X[source_cells, lgals9_idx]
            if hasattr(lgals9_expr, 'toarray'):
                lgals9_expr = lgals9_expr.toarray().flatten()
            mean_lgals9 = lgals9_expr.mean()
            n_source = source_cells.sum()
        else:
            mean_lgals9 = 0
            n_source = 0
        
        if target_cells.sum() > 0:
            havcr2_expr = adata.X[target_cells, havcr2_idx]
            if hasattr(havcr2_expr, 'toarray'):
                havcr2_expr = havcr2_expr.toarray().flatten()
            mean_havcr2 = havcr2_expr.mean()
            n_target = target_cells.sum()
        else:
            mean_havcr2 = 0
            n_target = 0
        
        combined_score = mean_lgals9 * mean_havcr2
        
        response = adata.obs[patient_mask]['Response'].iloc[0]
        
        patient_scores.append({
            'patient_ID': patient,
            'LGALS9_score': mean_lgals9,
            'HAVCR2_score': mean_havcr2,
            'Combined_score': combined_score,
            'n_source_cells': n_source,
            'n_target_cells': n_target,
            'Response': response
        })
    
    return pd.DataFrame(patient_scores)

print("\nCalculating patient-level scores...")
patient_scores = calculate_patient_score(adata_pre)

min_cells = 5
patient_scores_filtered = patient_scores[
    (patient_scores['n_source_cells'] >= min_cells) &
    (patient_scores['n_target_cells'] >= min_cells)
].copy()

print(f"\nPatients with ≥{min_cells} source and target cells: {len(patient_scores_filtered)}")
print(f"  Responders: {(patient_scores_filtered['Response'] == 'Responder').sum()}")
print(f"  Non-responders: {(patient_scores_filtered['Response'] == 'Non-responder').sum()}")

print("\n--- LGALS9 Score (Source Cells) ---")
print(patient_scores_filtered.groupby('Response')['LGALS9_score'].describe())

print("\n--- HAVCR2 Score (Exhausted CD8+ T-cells) ---")
print(patient_scores_filtered.groupby('Response')['HAVCR2_score'].describe())

print("\n--- Combined Score (LGALS9 × HAVCR2) ---")
print(patient_scores_filtered.groupby('Response')['Combined_score'].describe())

nr_lgals9 = patient_scores_filtered[patient_scores_filtered['Response'] == 'Non-responder']['LGALS9_score']
r_lgals9 = patient_scores_filtered[patient_scores_filtered['Response'] == 'Responder']['LGALS9_score']

nr_havcr2 = patient_scores_filtered[patient_scores_filtered['Response'] == 'Non-responder']['HAVCR2_score']
r_havcr2 = patient_scores_filtered[patient_scores_filtered['Response'] == 'Responder']['HAVCR2_score']

nr_combined = patient_scores_filtered[patient_scores_filtered['Response'] == 'Non-responder']['Combined_score']
r_combined = patient_scores_filtered[patient_scores_filtered['Response'] == 'Responder']['Combined_score']

print("\n=== STATISTICAL TESTS ===")

stat1, pval1 = stats.mannwhitneyu(nr_lgals9, r_lgals9, alternative='greater')
print(f"\nLGALS9: Non-responders > Responders")
print(f"  Mann-Whitney U: p = {pval1:.4e}")
print(f"  Fold change: {nr_lgals9.mean() / r_lgals9.mean():.2f}x")

stat2, pval2 = stats.mannwhitneyu(nr_havcr2, r_havcr2, alternative='greater')
print(f"\nHAVCR2: Non-responders > Responders")
print(f"  Mann-Whitney U: p = {pval2:.4e}")
print(f"  Fold change: {nr_havcr2.mean() / r_havcr2.mean():.2f}x")

stat3, pval3 = stats.mannwhitneyu(nr_combined, r_combined, alternative='greater')
print(f"\nCombined Score: Non-responders > Responders")
print(f"  Mann-Whitney U: p = {pval3:.4e}")
print(f"  Fold change: {nr_combined.mean() / r_combined.mean():.2f}x")

y_true = (patient_scores_filtered['Response'] == 'Non-responder').astype(int)
y_score = patient_scores_filtered['Combined_score']

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

print(f"\nBIOMARKER PERFORMANCE:")
print(f"AUC-ROC: {roc_auc:.3f}")

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Sensitivity: {tpr[optimal_idx]:.2%}")
print(f"Specificity: {(1-fpr[optimal_idx]):.2%}")
print(f"Balanced Accuracy: {((tpr[optimal_idx] + (1-fpr[optimal_idx]))/2):.2%}")

y_pred = (y_score >= optimal_threshold).astype(int)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, 
                            target_names=['Responder', 'Non-responder'],
                            digits=3))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(f"                Predicted R  Predicted NR")
print(f"Actual R        {cm[0,0]:>11}  {cm[0,1]:>12}")
print(f"Actual NR       {cm[1,0]:>11}  {cm[1,1]:>12}")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(patient_scores_filtered))
colors = ['#d95f02' if r == 'Non-responder' else '#7570b3' 
          for r in patient_scores_filtered['Response']]
ax1.bar(x_pos, patient_scores_filtered['LGALS9_score'], color=colors, alpha=0.7)
ax1.set_xlabel('Patient')
ax1.set_ylabel('LGALS9 Expression')
ax1.set_title('LGALS9 in Source Cells by Patient')
ax1.axhline(y=optimal_threshold, color='red', linestyle='--', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(x_pos, patient_scores_filtered['HAVCR2_score'], color=colors, alpha=0.7)
ax2.set_xlabel('Patient')
ax2.set_ylabel('HAVCR2 (TIM-3) Expression')
ax2.set_title('TIM-3 in Exhausted CD8+ T-cells by Patient')

ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(x_pos, patient_scores_filtered['Combined_score'], color=colors, alpha=0.7)
ax3.set_xlabel('Patient')
ax3.set_ylabel('Combined Score')
ax3.set_title('LGALS9 × TIM-3 Resistance Score')
ax3.axhline(y=optimal_threshold, color='red', linestyle='--', 
           label=f'Threshold: {optimal_threshold:.2f}')
ax3.legend()

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d95f02', alpha=0.7, label='Non-responder'),
                   Patch(facecolor='#7570b3', alpha=0.7, label='Responder')]
ax1.legend(handles=legend_elements, loc='upper right')

ax4 = fig.add_subplot(gs[1, 0])
patient_scores_filtered.boxplot(column='Combined_score', by='Response', ax=ax4,
                                patch_artist=True)
bp = ax4.boxplot([r_combined, nr_combined], 
                 labels=['Responder', 'Non-responder'],
                 patch_artist=True)
for patch, color in zip(bp['boxes'], ['#7570b3', '#d95f02']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('LGALS9 × TIM-3 Score')
ax4.set_xlabel('Response Status')
ax4.set_title(f'Score Distribution (p={pval3:.2e})')
plt.sca(ax4)
plt.xticks(rotation=0)
ax4.get_figure().suptitle('')

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(fpr, tpr, color='darkorange', lw=3, 
         label=f'LGALS9-TIM3 Score\n(AUC = {roc_auc:.3f})')
ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax5.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
           label=f'Optimal (Sens={tpr[optimal_idx]:.2f}, Spec={1-fpr[optimal_idx]:.2f})',
           zorder=5)
ax5.set_xlim([0.0, 1.0])
ax5.set_ylim([0.0, 1.05])
ax5.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax5.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
ax5.set_title('ROC Curve: Patient-Level Biomarker', fontsize=12, weight='bold')
ax5.legend(loc="lower right")
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
for response, color, marker in [('Non-responder', '#d95f02', 'o'), 
                                ('Responder', '#7570b3', 's')]:
    subset = patient_scores_filtered[patient_scores_filtered['Response'] == response]
    ax6.scatter(subset['LGALS9_score'], subset['HAVCR2_score'], 
               label=response, alpha=0.7, s=120, color=color, marker=marker,
               edgecolors='black', linewidths=1)
ax6.set_xlabel('LGALS9 (Source Cells)', fontsize=11)
ax6.set_ylabel('HAVCR2 (Exhausted CD8+)', fontsize=11)
ax6.set_title('Ligand-Receptor Correlation', fontsize=12, weight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

from scipy.stats import pearsonr
r_corr, p_corr = pearsonr(patient_scores_filtered['LGALS9_score'], 
                          patient_scores_filtered['HAVCR2_score'])
ax6.text(0.05, 0.95, f'r = {r_corr:.2f}, p = {p_corr:.2e}',
        transform=ax6.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('figures/patient_level_biomarker_complete.png', dpi=300, bbox_inches='tight')
print("\nSaved comprehensive biomarker analysis")
plt.show()

patient_scores_filtered.to_csv('patient_resistance_scores.csv', index=False)
print("Saved patient scores to 'patient_resistance_scores.csv'")

print("\n" + "="*70)
print("SUMMARY: LGALS9-TIM3 AS A PREDICTIVE BIOMARKER")
print("="*70)
print(f"Number of patients analyzed: {len(patient_scores_filtered)}")
print(f"  Responders: {(patient_scores_filtered['Response'] == 'Responder').sum()}")
print(f"  Non-responders: {(patient_scores_filtered['Response'] == 'Non-responder').sum()}")
print(f"\nBiomarker Performance:")
print(f"  AUC-ROC: {roc_auc:.3f}")
print(f"  Sensitivity: {tpr[optimal_idx]:.1%}")
print(f"  Specificity: {(1-fpr[optimal_idx]):.1%}")
print(f"  Balanced Accuracy: {((tpr[optimal_idx] + (1-fpr[optimal_idx]))/2):.1%}")
print(f"\nEffect Sizes:")
print(f"  LGALS9 fold change: {nr_lgals9.mean() / r_lgals9.mean():.2f}x (p={pval1:.2e})")
print(f"  HAVCR2 fold change: {nr_havcr2.mean() / r_havcr2.mean():.2f}x (p={pval2:.2e})")
print(f"  Combined score fold change: {nr_combined.mean() / r_combined.mean():.2f}x (p={pval3:.2e})")
print("="*70)
