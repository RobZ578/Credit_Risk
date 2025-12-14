"""
Target Engineering Module for Credit Risk Model
Creates proxy target variable 'is_high_risk' using RFM analysis and K-Means clustering.
Task 4: Create a proxy target variable for credit risk modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_proxy_target(raw_data_path, processed_df=None, snapshot_date="2025-12-01"):
    """
    Task 4: Create a proxy target variable 'is_high_risk' for credit risk.
    
    Parameters:
    - raw_data_path: str, path to raw transaction CSV
    - processed_df: pd.DataFrame, optional, your processed dataframe to merge target into
    - snapshot_date: str, reference date for recency calculation
    
    Returns:
    - rfm_df: pd.DataFrame with CustomerId, RFM, Cluster, and is_high_risk
    - merged_df: pd.DataFrame with processed_df merged with is_high_risk (if processed_df is provided)
    """
    
    logger.info("Starting Task 4: Proxy Target Variable Engineering")
    
    # 1. Load raw data
    logger.info(f"Loading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    logger.info(f"Raw data shape: {df.shape}")
    
    # 2. Convert date column
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    snapshot_date = pd.to_datetime(snapshot_date)
    logger.info(f"Using snapshot date: {snapshot_date}")
    
    # 3. Calculate RFM metrics
    logger.info("Calculating RFM metrics...")
    rfm_df = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'TransactionAmount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'CustomerId': 'Frequency',
        'TransactionAmount': 'Monetary'
    }).reset_index()
    
    logger.info(f"RFM calculated for {len(rfm_df)} customers")
    
    # 4. Scale features
    logger.info("Scaling RFM features...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    
    # 5. K-Means clustering with 3 clusters
    logger.info("Applying K-Means clustering with 3 clusters...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 6. Identify high-risk cluster
    logger.info("Identifying high-risk cluster...")
    cluster_risk = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # High-risk = High Recency (inactive), Low Frequency, Low Monetary
    high_risk_cluster = cluster_risk.sort_values(
        ['Recency', 'Frequency', 'Monetary'], 
        ascending=[False, True, True]
    ).index[0]
    
    logger.info(f"High-risk cluster identified: Cluster {high_risk_cluster}")
    
    # 7. Create binary target variable
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    
    # Log target distribution
    target_counts = rfm_df['is_high_risk'].value_counts()
    logger.info(f"Target distribution: {target_counts.to_dict()}")
    logger.info(f"High-risk percentage: {(target_counts.get(1, 0) / len(rfm_df) * 100):.1f}%")
    
    # 8. Merge with processed dataframe if provided
    if processed_df is not None:
        logger.info("Merging target variable with processed data...")
        merged_df = processed_df.merge(
            rfm_df[['CustomerId', 'is_high_risk']], 
            on='CustomerId', 
            how='left'
        )
        logger.info(f"Merged data shape: {merged_df.shape}")
        return rfm_df, merged_df
    
    logger.info("Task 4 completed successfully")
    return rfm_df


def save_target_results(rfm_df, output_path="../data/processed/"):
    """
    Save the RFM and target data to CSV files.
    
    Parameters:
    - rfm_df: DataFrame with RFM metrics and target
    - output_path: Directory to save results
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save RFM data with target
    rfm_path = f"{output_path}rfm_with_target.csv"
    rfm_df.to_csv(rfm_path, index=False)
    logger.info(f"RFM data saved to: {rfm_path}")
    
    # Save cluster summary
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': ['mean', 'std', 'count'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std'],
        'is_high_risk': 'mean'
    }).round(2)
    
    summary_path = f"{output_path}cluster_summary.csv"
    cluster_summary.to_csv(summary_path)
    logger.info(f"Cluster summary saved to: {summary_path}")
    
    return rfm_path, summary_path