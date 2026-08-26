"""
Statistical Analysis & Reproduction Script for the Presence Factor Scale (PFS)
Anonymized Supplementary Material for Peer Review

Dependencies:
    pip install pandas numpy scipy pingouin
"""

import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

def run_benchmark1_analysis():
    print("=" * 60)
    print("BENCHMARK 1: CONTROLLED LABORATORY BENCHMARK (N = 123)")
    print("=" * 60)
    
    df1 = pd.read_csv("benchmark1_internal_validation.csv")
    
    # Calculate mean PFS across T1 and T2
    df1['PFS_Mean_Score'] = (df1['PFS_T1'] + df1['PFS_T2']) / 2.0
    
    # Pearson correlation (PFS vs Empirical SUS)
    r_val, p_val = stats.pearsonr(df1['PFS_Mean_Score'], df1['SUS_Empirical_Score'])
    
    # Absolute Mean Difference (M_diff)
    m_diff = np.mean(np.abs(df1['PFS_Mean_Score'] - df1['SUS_Empirical_Score']))
    sd_diff = np.std(np.abs(df1['PFS_Mean_Score'] - df1['SUS_Empirical_Score']), ddof=1)
    
    # Intra-rater stability (T1 vs T2)
    test_retest_diff = np.mean(np.abs(df1['PFS_T1'] - df1['PFS_T2']))
    
    print(f"Sample Size (Participants): 123 across 5 VEs")
    print(f"Pearson Correlation (r):     {r_val:.4f} (p = {p_val:.4e})")
    print(f"Mean Absolute Error (M_diff): {m_diff:.4f} (SD = {sd_diff:.4f})")
    print(f"Test-Retest Mean Diff (T1-T2): {test_retest_diff:.4f}")
    print()


def run_benchmark2_analysis():
    print("=" * 60)
    print("BENCHMARK 2: ECOLOGICAL MULTI-RATER BENCHMARK (N = 17 VEs)")
    print("=" * 60)
    
    df2 = pd.read_csv("benchmark2_external_validation.csv")
    
    # 1. Pearson Correlation (PFS Mean vs Subjective Normalized Score)
    r_val, p_val = stats.pearsonr(df2['PFS_Mean'], df2['Subj_Normalized_Score'])
    
    # 2. Mean Absolute Discrepancy (M_diff)
    m_diff = np.mean(np.abs(df2['PFS_Mean'] - df2['Subj_Normalized_Score']))
    sd_diff = np.std(np.abs(df2['PFS_Mean'] - df2['Subj_Normalized_Score']), ddof=1)
    
    # 3. Inter-Rater Reliability (ICC) Calculation
    # Reshaping data into long format for Pingouin ICC
    raters_df = df2[['VE_ID', 'R1', 'R2', 'R3', 'R4']]
    long_df = pd.melt(
        raters_df, 
        id_vars=['VE_ID'], 
        value_vars=['R1', 'R2', 'R3', 'R4'],
        var_name='Rater', 
        value_name='Score'
    )
    
    icc_results = pg.intraclass_corr(
        data=long_df, 
        targets='VE_ID', 
        raters='Rater', 
        ratings='Score'
    ).round(4)
    
    # Extract ICC(2,1) and ICC(2,k)
    icc2_single = icc_results.loc[icc_results['Type'] == 'ICC2', 'ICC'].values[0]
    icc2_average = icc_results.loc[icc_results['Type'] == 'ICC2k', 'ICC'].values[0]
    
    print(f"Corpus Size (VEs):           17 external systems")
    print(f"Pearson Correlation (r):     {r_val:.4f} (p = {p_val:.4e})")
    print(f"Mean Absolute Error (M_diff): {m_diff:.4f} (SD = {sd_diff:.4f})")
    print(f"Inter-Rater ICC(2,1) Single:  {icc2_single:.4f}")
    print(f"Inter-Rater ICC(2,4) Average: {icc2_average:.4f}")
    print("\nComplete ICC Table:")
    print(icc_results[['Type', 'Description', 'ICC', 'F', 'pval', 'CI95%']].to_string(index=False))
    print()

if __name__ == "__main__":
    run_benchmark1_analysis()
    run_benchmark2_analysis()