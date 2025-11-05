import pandas as pd

results_df = pd.read_csv('liana_merged_results.csv')
resistance_signals = results_df[
    (results_df['specificity_rank_NR'] <= 0.1) &
    (results_df['specificity_rank_R'] > 0.3)
].copy()

print("Available columns:")
print(results_df.columns.tolist())

print("\n--- Interactions strong in BOTH groups (sanity check) ---")
both_strong = results_df[
    (results_df['specificity_rank_NR'] <= 0.1) &
    (results_df['specificity_rank_R'] <= 0.1)
].sort_values('specificity_rank_NR')
print(both_strong[['source', 'target', 'ligand_complex', 'receptor_complex', 
                   'specificity_rank_NR', 'specificity_rank_R']].head(20))

print("\n--- Distribution of Responder ranks for top NR interactions ---")
top_nr = results_df[results_df['specificity_rank_NR'] <= 0.1]
print(top_nr['specificity_rank_R'].describe())
print(f"Number with rank = 1.0: {(top_nr['specificity_rank_R'] == 1.0).sum()}")
print(f"Number with rank < 1.0: {(top_nr['specificity_rank_R'] < 1.0).sum()}")
