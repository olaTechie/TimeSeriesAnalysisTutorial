"""
Simplified Data Processing Module for Kampala TB Analysis
This module handles data loading and preprocessing for the simplified single-city TB analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedDataProcessor:
    """Handle data loading, cleaning, and preprocessing for Kampala TB data"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize SimplifiedDataProcessor
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path('.')
        self.data = None
        self.processed_data = None
        logger.info(f"SimplifiedDataProcessor initialized with data directory: {self.data_dir}")
    
    def load_data(self, filepath: str = 'final_data.csv', validate: bool = True) -> pd.DataFrame:
        """
        Load and process the merged Kampala TB data
        
        Args:
            filepath: Path to the merged CSV file
            validate: Whether to validate data quality
            
        Returns:
            DataFrame with processed data
        """
        logger.info(f"Loading data from {filepath}")
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records")
            
            # Convert date column
            # FIX 1: Added dayfirst=True to correctly parse DD/MM/YYYY dates
            df['date'] = pd.to_datetime(df['ds'], dayfirst=True)
            df['year_month'] = df['date'].dt.to_period('M')
            
            # Validate required columns
            required_cols = ['pm2_5', 'TB', 'avgtemp', 'avghumidity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
            
            # Handle missing values
            df['pm2_5'] = df['pm2_5'].fillna(df['pm2_5'].mean())
            df['TB'] = df['TB'].fillna(0)
            df['avgtemp'] = df['avgtemp'].fillna(df['avgtemp'].mean())
            df['avghumidity'] = df['avghumidity'].fillna(df['avghumidity'].mean())
            
            # Add Kampala as county (for compatibility with existing code)
            df['county'] = 'Kampala'
            
            # Rename columns for compatibility
            df['avg_pm2_5_calibrated'] = df['pm2_5']
            df['total_tb'] = df['TB']
            df['temperature'] = df['avgtemp']
            df['humidity'] = df['avghumidity']
            
            # Add placeholder for pneumonia (set to 0)
            df['total_pneumonia'] = 0
            
            # Sort by date
            df = df.sort_values('date')
            
            if validate:
                self._validate_data(df)
            
            self.data = df
            logger.info(f"Data processing complete: {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def add_lag_features(self, lags: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        """
        Add lag features for time series analysis
        
        Args:
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features added
        """
        if self.data is None:
            raise ValueError("Load data first before adding features")
        
        df = self.data.copy()
        
        # Sort by date to ensure correct lag calculation
        df = df.sort_values('date')
        
        # Add lag features for PM2.5 and TB
        for lag in lags:
            df[f'pm25_lag_{lag}'] = df['pm2_5'].shift(lag)
            df[f'tb_lag_{lag}'] = df['TB'].shift(lag)
            df[f'temp_lag_{lag}'] = df['avgtemp'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['avghumidity'].shift(lag)
        
        # Add rolling averages
        for window in [7, 14, 30]:  # Weekly, bi-weekly, monthly
            df[f'pm25_ma_{window}'] = df['pm2_5'].rolling(window=window, min_periods=1).mean()
            df[f'tb_ma_{window}'] = df['TB'].rolling(window=window, min_periods=1).mean()
        
        # Add seasonal features
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # FIX 2: Added 'month' and 'week_of_year' columns needed for feature engineering
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Season based on typical Uganda patterns
        df['season'] = df['month'].apply(lambda x: 'dry' if x in [12, 1, 2, 6, 7, 8] else 'wet')
        
        self.processed_data = df
        logger.info(f"Added lag features and rolling averages")
        
        return df
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the data
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            raise ValueError("Load data first")
        
        summary = {
            'total_records': len(self.data),
            'date_range': f"{self.data['date'].min()} to {self.data['date'].max()}",
            'pm25_statistics': {
                'mean': self.data['pm2_5'].mean(),
                'std': self.data['pm2_5'].std(),
                'min': self.data['pm2_5'].min(),
                'max': self.data['pm2_5'].max(),
                'median': self.data['pm2_5'].median()
            },
            'tb_statistics': {
                'total_cases': self.data['TB'].sum(),
                'mean': self.data['TB'].mean(),
                'std': self.data['TB'].std(),
                'min': self.data['TB'].min(),
                'max': self.data['TB'].max()
            },
            'temperature_statistics': {
                'mean': self.data['avgtemp'].mean(),
                'std': self.data['avgtemp'].std(),
                'min': self.data['avgtemp'].min(),
                'max': self.data['avgtemp'].max()
            },
            'humidity_statistics': {
                'mean': self.data['avghumidity'].mean(),
                'std': self.data['avghumidity'].std(),
                'min': self.data['avghumidity'].min(),
                'max': self.data['avghumidity'].max()
            },
            'missing_values': self.data.isnull().sum().to_dict(),
            'correlation_pm25_tb': self.data['pm2_5'].corr(self.data['TB'])
        }
        
        return summary
    
    def prepare_for_modeling(self, target: str = 'TB', 
                           feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning modeling
        
        Args:
            target: Target variable for prediction
            feature_cols: List of feature columns to use
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.processed_data is None:
            self.add_lag_features()
        
        df = self.processed_data.copy()
        
        # Default features if not specified
        if feature_cols is None:
            feature_cols = [
                'pm2_5', 'avgtemp', 'avghumidity',
                'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3',
                'tb_lag_1', 'tb_lag_2', 'tb_lag_3',
                'pm25_ma_7', 'pm25_ma_14',
                'week_of_year', 'quarter'
            ]
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Remove rows with NaN values
        df_clean = df.dropna(subset=available_features + [target])
        
        X = df_clean[available_features]
        y = df_clean[target]
        
        logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
        
        return X, y
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate data quality"""
        # Check for negative values
        if (df['pm2_5'] < 0).any():
            logger.warning("Negative PM2.5 values detected, setting to 0")
            df.loc[df['pm2_5'] < 0, 'pm2_5'] = 0
        
        if (df['TB'] < 0).any():
            logger.warning("Negative TB values detected, setting to 0")
            df.loc[df['TB'] < 0, 'TB'] = 0
        
        # Check for extreme values
        pm25_upper = df['pm2_5'].quantile(0.99) * 3
        if (df['pm2_5'] > pm25_upper).any():
            logger.warning(f"Extreme PM2.5 values (>{pm25_upper:.1f}) detected")
        
        # Check data completeness
        date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='W')
        missing_weeks = len(date_range) - len(df)
        if missing_weeks > 0:
            logger.warning(f"{missing_weeks} weeks missing from data")
    
    def save_processed_data(self, output_path: str = 'processed_kampala_tb_data.csv'):
        """
        Save processed data to CSV
        
        Args:
            output_path: Path to save processed data
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        else:
            logger.warning("No processed data to save")
    
    def get_intervention_baseline(self) -> Dict:
        """
        Get baseline metrics for intervention analysis
        
        Returns:
            Dictionary with baseline metrics
        """
        if self.data is None:
            raise ValueError("Load data first")
        
        return {
            'baseline_pm25': self.data['pm2_5'].mean(),
            'baseline_tb': self.data['TB'].mean(),
            'total_tb_cases': self.data['TB'].sum(),
            'pm25_std': self.data['pm2_5'].std(),
            'tb_std': self.data['TB'].std(),
            'correlation': self.data['pm2_5'].corr(self.data['TB'])
        }