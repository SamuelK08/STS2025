 Validate these findings in the actual expression data
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

adata = sc.read('processed_melanoma_data_fortified.h5ad')

key_genes = ['LGALS9', 'HMGB1', 'HAVCR2', 'CD47', 'SIRPG', 'BTLA', 'CD247']

pre_mask = adata.obs['Time'].str.contains('Pre', na=False)
adata_pre = adata[pre_mask].copy()

for gene in key_genes:
    if gene in adata_pre.var_names:
        sc.pl.violin(adata_pre, gene, groupby='Response', 
                    save=f'_{gene}_response.pdf')

print("\n=== ENDOTHELIAL CHECKPOINT EXPRESSION ===")
endo_mask = adata_pre.obs['cell_type'] == 'Endothelial'
adata_endo = adata_pre[endo_mask].copy()

for gene in ['LGALS9', 'HMGB1', 'BTLA']:
    if gene in adata_endo.var_names:
        gene_idx = adata_endo.var_names.get_loc(gene)
        expression = adata_endo.X[:, gene_idx]
        
        if hasattr(expression, 'toarray'):
            expression = expression.toarray().flatten()
        
        expr_df = pd.DataFrame({
            'expression': expression,
            'Response': adata_endo.obs['Response'].values
        })
        
        print(f"\n{gene}:")
        print(expr_df.groupby('Response')['expression'].describe())
        
        from scipy import stats
        nr_expr = expr_df[expr_df['Response'] == 'Non-responder']['expression']
        r_expr = expr_df[expr_df['Response'] == 'Responder']['expression']
        
        statistic, pvalue = stats.mannwhitneyu(nr_expr, r_expr, alternative='two-sided')
        print(f"Mann-Whitney U test: p-value = {pvalue:.4e}")
        
        nr_mean = nr_expr[nr_expr > 0].mean() if (nr_expr > 0).any() else 0
        r_mean = r_expr[r_expr > 0].mean() if (r_expr > 0).any() else 0
        if r_mean > 0:
            fold_change = nr_mean / r_mean
            print(f"Fold change (NR/R): {fold_change:.2f}x")

print("\n\n=== EXPRESSION BY CELL TYPE ===")
for gene in ['LGALS9', 'HMGB1', 'HAVCR2']:
    if gene in adata_pre.var_names:
        print(f"\n{gene} expression by cell type and response:")
        
        gene_idx = adata_pre.var_names.get_loc(gene)
        expression = adata_pre.X[:, gene_idx]
        if hasattr(expression, 'toarray'):
            expression = expression.toarray().flatten()
        
        expr_df = pd.DataFrame({
            'expression': expression,
            'cell_type': adata_pre.obs['cell_type'].values,
            'Response': adata_pre.obs['Response'].values
        })
        
        summary = expr_df.groupby(['cell_type', 'Response'])['expression'].agg(['mean', 'count'])
        summary = summary.unstack(fill_value=0)
        print(summary)

print("\n\nGenerating expression heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for idx, gene in enumerate(['LGALS9', 'HMGB1', 'HAVCR2']):
    if gene in adata_pre.var_names:
        gene_idx = adata_pre.var_names.get_loc(gene)
        expression = adata_pre.X[:, gene_idx]
        if hasattr(expression, 'toarray'):
            expression = expression.toarray().flatten()
        
        expr_df = pd.DataFrame({
            'expression': expression,
            'cell_type': adata_pre.obs['cell_type'].values,
            'Response': adata_pre.obs['Response'].values
        })
        
        pivot_data = expr_df.groupby(['cell_type', 'Response'])['expression'].mean().unstack()
        
        ax = axes[idx]
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'Mean Expression'})
        ax.set_title(f'{gene} Expression', fontsize=14, weight='bold')
        ax.set_ylabel('Cell Type')
        ax.set_xlabel('Response Status')

plt.tight_layout()
plt.savefig('figures/key_genes_expression_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved expression heatmap to figures/key_genes_expression_heatmap.png")
plt.show()

print("\n\n=== TIM-3 PATHWAY ANALYSIS ===")
tim3_genes = ['LGALS9', 'HMGB1', 'HAVCR2']
available_genes = [g for g in tim3_genes if g in adata_pre.var_names]

if len(available_genes) > 0:
    pathway_expr = []
    for gene in available_genes:
        gene_idx = adata_pre.var_names.get_loc(gene)
        expression = adata_pre.X[:, gene_idx]
        if hasattr(expression, 'toarray'):
            expression = expression.toarray().flatten()
        pathway_expr.append(expression)
    
    pathway_score = np.mean(pathway_expr, axis=0)
    
    adata_pre.obs['TIM3_pathway_score'] = pathway_score
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    score_df = pd.DataFrame({
        'TIM3_score': pathway_score,
        'cell_type': adata_pre.obs['cell_type'].values,
        'Response': adata_pre.obs['Response'].values
    })
    
    cell_order = score_df.groupby('cell_type')['TIM3_score'].mean().sort_values(ascending=False).index
    
    sns.violinplot(data=score_df, x='cell_type', y='TIM3_score', hue='Response',
                  order=cell_order, palette={'Non-responder': '#d95f02', 'Responder': '#7570b3'},
                  ax=ax)
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('TIM-3 Pathway Score', fontsize=12)
    ax.set_title('TIM-3 Pathway Activity Across Cell Types', fontsize=14, weight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/TIM3_pathway_score.png', dpi=300, bbox_inches='tight')
    print("Saved TIM-3 pathway score plot")
    plt.show()
    
    print("\nTIM-3 pathway score by response status:")
    print(score_df.groupby('Response')['TIM3_score'].describe())
    
    from scipy import stats
    nr_scores = score_df[score_df['Response'] == 'Non-responder']['TIM3_score']
    r_scores = score_df[score_df['Response'] == 'Responder']['TIM3_score']
    statistic, pvalue = stats.mannwhitneyu(nr_scores, r_scores, alternative='greater')
    print(f"\nMann-Whitney U test (NR > R): p-value = {pvalue:.4e}")

print("\n\nAnalysis complete!")
