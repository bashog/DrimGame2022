"""
This file contains the columns choosed during the prepocessing
The columns depends on the chronique type
"""

# TOTALE
col_totale_corr = ['p95_3', 'mean_7', 'p95_6', 'median_2', 'CD_MOD_HABI_1', 'PIB', 'p90_6', 'p95_2', 'p90_2', 'p90_3', 'p75_7', 'p90_7', 'mean_1', 'mean_6', 'p95_1', 'Tx_cho', 'mean_2']
col_2_low_var = ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2', 'p90_2',
       'p95_2', 'mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3',
       'p90_3', 'p95_3', 'mean_4', 'median_4', 'p75_4', 'p90_4', 'p95_4',
       'mean_5', 'p5_5', 'p10_5', 'p25_5', 'p75_5', 'p90_5', 'p95_5', 'PIB',
       'Inflation']
col_totale_kbest = ['mean_1', 'p90_1', 'p95_1', 'mean_2', 'median_2', 'p90_2', 'p95_2',
       'p90_3', 'p95_3', 'mean_6', 'p90_6', 'p95_6', 'mean_7', 'median_7',
       'p25_7', 'p75_7', 'p90_7', 'CD_MOD_HABI_1', 'PIB', 'Tx_cho']
col_totale_recur = ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2', 'p90_2',
       'p95_2', 'mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3',
       'p90_3', 'p95_3', 'mean_4', 'median_4', 'p5_4', 'p25_4', 'p75_4',
       'p90_4', 'p95_4', 'p10_7', 'p25_7', 'p90_7', 'median_8', 'p25_8',
       'Tx_cho']
col_totale_sfm = ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2', 'p90_2',
       'p95_2', 'median_3', 'p5_3', 'p25_3', 'p90_3', 'p95_3', 'mean_4',
       'median_4', 'p25_4', 'p75_4', 'p90_4', 'p95_4']
col_totale_tree = ['mean_1', 'p25_1', 'p90_1', 'mean_2', 'p90_2', 'p75_4', 'p90_4',
       'p75_5', 'p90_5', 'mean_6', 'p90_6', 'p95_6', 'mean_7', 'median_7',
       'p5_7', 'p75_7', 'p90_7', 'p25_8', 'p90_8', 'p95_8', 'CD_TY_CLI_RCI_1',
       'CD_MOD_HABI_1', 'CD_QUAL_VEH_1', 'PIB', 'Tx_cho']
col_totale_seq_for = ['p5_2', 'p10_2', 'p25_2', 'p75_2', 'p95_2', 'p5_3', 'p10_3', 'p5_5',
       'p95_5', 'median_6', 'p5_6', 'p10_6', 'p25_6', 'p75_6', 'p10_7',
       'p25_7', 'p5_8', 'p75_8', 'CD_MOD_HABI_1', 'PIB']
col_totale_seq_bac = ['median_2', 'p5_2', 'p25_2', 'p90_2', 'p10_3', 'p25_3', 'p75_3',
       'p90_3', 'p95_3', 'median_4', 'p90_4', 'p95_4', 'mean_5', 'p5_5',
       'p10_5', 'p25_5', 'p75_5', 'p90_5', 'p95_5', 'PIB']

# CHR8
col_8_corr = ['p95_5', 'p25_1', 'p25_7', 'mean_5', 'p95_4', 'p75_1', 'mean_1', 'CD_QUAL_VEH_1', 'mean_7', 'CD_ETA_CIV_1', 'mean_4', 'mean_8', 'p5_7', 'median_1', 'p25_5', 'p90_1', 'p95_1', 'PIB', 'p75_4', 'p95_8', 'p75_7', 'p90_4', 'median_7']
col_8_low_var = ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2', 'p90_2',
       'p95_2', 'mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3',
       'p90_3', 'p95_3', 'mean_4', 'median_4', 'p25_4', 'p75_4', 'p90_4',
       'p95_4', 'mean_5', 'p5_5', 'p10_5', 'p25_5', 'p75_5', 'p90_5', 'p95_5',
       'PIB', 'Inflation']
col_8_kbest = ['mean_1', 'median_1', 'p25_1', 'p75_1', 'p90_1', 'p95_1', 'mean_4',
       'p75_4', 'p90_4', 'mean_5', 'p25_5', 'mean_7', 'p5_7', 'mean_8',
       'median_8', 'CD_ETA_CIV_1', 'CD_MOD_HABI_1', 'CD_QUAL_VEH_1', 'PIB',
       'Tx_cho']
col_8_recur = ['mean_1', 'median_1', 'p5_1', 'p10_1', 'p25_1', 'p75_1', 'p90_1',
       'p95_1', 'mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2',
       'p90_2', 'p95_2', 'mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3',
       'p75_3', 'p90_3', 'p95_3', 'mean_4', 'median_4', 'p5_4', 'p10_4',
       'p25_4', 'p75_4', 'p90_4', 'p95_4', 'mean_5', 'p5_5', 'p75_5', 'p5_6',
       'p25_6', 'mean_7', 'median_7', 'p5_7', 'p10_7', 'p25_7', 'p75_7',
       'p90_7', 'p95_7', 'mean_8', 'median_8', 'p5_8', 'p25_8', 'p75_8',
       'p90_8', 'p95_8', 'CD_TY_CLI_RCI_1', 'CD_ETA_CIV_1', 'CD_MOD_HABI_1',
       'CD_PROF_1', 'CD_PROF_2', 'CD_QUAL_VEH_1', 'Inflation', 'Tx_cho']
col_8_sfm = ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2', 'p95_2',
       'mean_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3', 'p95_3', 'median_4',
       'p10_4', 'p25_4', 'p75_4', 'p90_4', 'p95_4', 'Tx_cho']
col_8_tree = ['mean_1', 'p75_1', 'p90_1', 'p95_2', 'p25_3', 'mean_4', 'p75_4',
       'p90_4', 'mean_5', 'p25_5', 'p95_5', 'mean_7', 'p5_7', 'p25_7', 'p90_7',
       'mean_8', 'median_8', 'CD_TY_CLI_RCI_1', 'CD_ETA_CIV_1',
       'CD_QUAL_VEH_1', 'PIB']
col_8_seq_for = ['p5_2', 'p10_2', 'p25_2', 'median_3', 'p90_3', 'mean_4', 'p5_4',
       'p25_4', 'p75_4', 'p95_4', 'mean_5', 'p5_5', 'p10_6', 'p25_6', 'p75_6',
       'p90_6', 'p95_6', 'p10_8', 'p25_8', 'CD_MOD_HABI_1']
col_8_seq_bac = ['mean_2', 'median_2', 'p10_2', 'p25_2', 'p75_2', 'p95_2', 'p5_3',
       'p10_3', 'p75_3', 'p90_3', 'mean_4', 'median_4', 'p25_4', 'p75_4',
       'p90_4', 'p95_4', 'mean_5', 'p5_5', 'p25_5', 'p95_5']

# CHR2
col_2_corr = ['median_1']
col_2_low_var = ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2', 'p90_2',
       'p95_2', 'mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3',
       'p90_3', 'p95_3', 'mean_4', 'median_4', 'p5_4', 'p10_4', 'p25_4',
       'p75_4', 'p90_4', 'p95_4', 'mean_5', 'p5_5', 'p10_5', 'p25_5', 'p75_5',
       'p90_5', 'p95_5', 'PIB', 'Inflation']
col_2_kbest = ['mean_1', 'median_1', 'p25_1', 'p75_1', 'p95_1', 'p5_3', 'mean_4',
       'mean_5', 'p75_5', 'mean_8', 'p5_8', 'p10_8', 'p25_8', 'p75_8', 'p90_8',
       'p95_8', 'CD_TY_CLI_RCI_1', 'CD_QUAL_VEH_1', 'PIB', 'Inflation']
col_2_recur = ['mean_2', 'median_2', 'p75_2', 'mean_3', 'median_3', 'p5_3', 'p10_3',
       'p25_3', 'p75_3', 'p90_3', 'p95_3', 'mean_4', 'median_4', 'p5_4',
       'p10_4', 'p25_4', 'p75_4', 'p90_4', 'p95_4', 'mean_7', 'median_7',
       'p5_7', 'p25_7', 'p75_7', 'p90_7', 'p95_7', 'mean_8', 'median_8',
       'p5_8', 'p10_8', 'p75_8', 'p90_8', 'p95_8', 'CD_ETA_CIV_1', 'CD_PROF_2',
       'CD_QUAL_VEH_1', 'Inflation', 'Tx_cho']
col_2_sfm = ['mean_2', 'median_2', 'p25_2', 'mean_3', 'median_3', 'p10_3', 'p25_3',
       'p75_3', 'p90_3', 'p95_3', 'mean_4', 'median_4', 'p5_4', 'p10_4',
       'p75_4', 'p90_4', 'Inflation']
col_2_tree = ['median_1', 'p5_1', 'p10_1', 'p25_1', 'p5_3', 'p5_4', 'p95_5',
       'median_7', 'p10_7', 'p90_7', 'median_8', 'p5_8', 'p25_8', 'p90_8',
       'p95_8', 'CD_TY_CLI_RCI_1', 'CD_PROF_1', 'CD_QUAL_VEH_1', 'PIB',
       'Inflation', 'Tx_cho']
col_2_seq_for = ['p5_2', 'p10_2', 'p90_2', 'median_3', 'p25_3', 'p75_3', 'p5_4',
       'median_6', 'p5_6', 'p10_6', 'p25_6', 'p75_6', 'p90_6', 'p95_6',
       'p95_7', 'p5_8', 'p10_8', 'p90_8', 'CD_TY_CLI_RCI_1', 'CD_QUAL_VEH_1']
col_2_seq_bac = ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p95_2', 'mean_3',
       'p5_3', 'p25_3', 'p75_3', 'p95_3', 'p5_4', 'p90_4', 'p95_4', 'mean_5',
       'p10_5', 'p25_5', 'p75_5', 'p95_5', 'PIB']
