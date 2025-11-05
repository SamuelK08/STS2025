print("Step 1: Loading libraries and data...")

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(42)

sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
pd.set_option('display.max_columns', 50)

if not os.path.exists('figures'):
    os.makedirs('figures')
sc.settings.figdir = 'figures/'

expr_matrix_path = 'GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz'
metadata_path = 'GSE120575_patient_ID_single_cells.txt.gz'

print("Loading data with corrected parsing (this may take a while)...")
expr_df = pd.read_csv(expr_matrix_path, sep='\t', header=0, skiprows=[1], index_col=0)
metadata_df = pd.read_csv(metadata_path, sep='\t', skiprows=19, header=0, index_col='title', encoding='latin-1')
metadata_df = metadata_df.rename(columns={
    'characteristics: patinet ID (Pre=baseline; Post= on treatment)': 'Time',
    'characteristics: response': 'Response'
})
metadata_df['patient_ID'] = metadata_df['Time'].str.split('_').str[1]
print("Metadata successfully loaded and cleaned.")

adata = sc.AnnData(expr_df.T)
adata.obs = adata.obs.join(metadata_df)

print("\nValidating metadata merge...")
adata = adata[~adata.obs['Time'].isnull()].copy()
required_cols = ['Time', 'Response', 'patient_ID']
assert all(col in adata.obs.columns for col in required_cols), "ERROR: Key metadata columns are missing!"
print("SUCCESS: Metadata loaded and validated.")

print("\n--- Dataset Summary ---")
print(f"Total cells after merging: {adata.n_obs}")
print("Cell counts by timepoint:\n", adata.obs['Time'].value_counts().head())
print("\nCell counts by response:\n", adata.obs['Response'].value_counts())
print("\nCell counts per patient:\n", adata.obs['patient_ID'].value_counts().head())
print("-----------------------\n")

print("SAVING CHECKPOINT: Saving the initial loaded data object for fast reloading...")
adata.write('initial_melanoma_data.h5ad')
print("Successfully saved initial data to 'initial_melanoma_data.h5ad'.\n")

print("Step 2: Performing Quality Control...")
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save="_qc_violin.png")
min_genes, max_mito_pct = 200, 20.0
print(f"Applying filters: min_genes > {min_genes}, pct_counts_mt < {max_mito_pct}")
sc.pp.filter_cells(adata, min_genes=min_genes)
adata = adata[adata.obs.pct_counts_mt < max_mito_pct, :]
print("Filtering genes expressed in fewer than 10 cells.")
sc.pp.filter_genes(adata, min_cells=10)
print(f"Data after QC: {adata.n_obs} cells and {adata.n_vars} genes remaining.")

print("\nStep 3: Normalization and Feature Selection...")
sc.pp.log1p(adata)
print("Finding highly variable genes using 'seurat' flavor...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
adata = adata[:, adata.var.highly_variable]

print("\nStep 4: Dimensionality Reduction and Clustering...")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack', random_state=42)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
print("Clustering cells with Leiden algorithm (resolution=0.8)...")
sc.tl.leiden(adata, resolution=0.8, random_state=42)
sc.tl.umap(adata, random_state=42)

print("\nStep 5: Cell Type Annotation WITH VALIDATION...")
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

print("Validating clusters with canonical marker gene dot plot...")
marker_genes_dict = {
    'T-cells': ['CD3D', 'CD3E'],
    'CD8 T-cells': ['CD8A', 'CD8B'],
    'CD4 T-cells': ['CD4', 'IL7R'],
    'Tregs': ['FOXP3', 'IL2RA'],
    'Exhausted T': ['PDCD1', 'CTLA4', 'LAG3', 'TOX'],
    'NK cells': ['NKG7', 'GNLY'],
    'B-cells': ['MS4A1', 'CD19'],
    'Plasma cells': ['JCHAIN', 'MZB1'],
    'Macrophages': ['LYZ', 'CD68', 'C1QA'],
    'Endothelial': ['PECAM1', 'VWF'],
    'Fibroblasts': ['DCN', 'COL1A1'],
    'Melanoma': ['MLANA', 'PMEL', 'MITF']
}

sc.pl.dotplot(adata, marker_genes_dict, groupby='leiden', dendrogram=True, save="_marker_validation.png")
print("Saved marker validation dot plot. Annotations will be based on this.")

cluster_to_celltype = {
    '0': 'CD8+ T-cells', '1': 'Malignant', '2': 'Macrophages',
    '3': 'Cancer-Associated Fibroblasts (CAFs)', '4': 'T-cells', '5': 'Malignant',
    '6': 'Exhausted CD8+ T-cells', '7': 'Malignant', '8': 'NK cells',
    '9': 'Endothelial', '10': 'Regulatory T-cells (Tregs)', '11': 'B-cells',
    '12': 'CD4+ T-cells', '13': 'T-cells', '14': 'Plasma cells'
}
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_to_celltype).fillna('Unknown')
print("\nRefined cell type counts:\n", adata.obs['cell_type'].value_counts())

print("\nStep 6: Biological Visualization & Analysis...")
print("Generating UMAPs colored by key biological variables...")
sc.pl.umap(adata, color=['cell_type', 'Response', 'Time'], save="_biological_umaps.png")
print("Analyzing cell type proportions between Responders and Non-Responders...")
composition_df = pd.crosstab(adata.obs['Response'], adata.obs['cell_type'], normalize='index')
composition_df.plot.bar(stacked=True, figsize=(12, 7), colormap='tab20')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.title('Cell Type Proportions by Response Status')
plt.ylabel('Fraction of Cells')
plt.tight_layout()
plt.savefig('figures/composition_by_response.png')
print("Saved cell type composition plot.")

print("\nStep 7: Saving Final Annotated Object...")
adata.write('processed_melanoma_data_fortified.h5ad')
print("\nPhase 2 Complete!")
print("Successfully saved fortified data object to 'processed_melanoma_data_fortified.h5ad'.")
