"""
Data Processing Module for Air Quality Epidemiological Analysis
This module handles all data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize DataProcessor
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path('.')
        self.pm25_data = None
        self.health_data = {}
        self.processed_data = None
        logger.info(f"DataProcessor initialized with data directory: {self.data_dir}")
    
    def load_pm25_data(self, filepath: str, validate: bool = True) -> pd.DataFrame:
        """
        Load and process PM2.5 exposure data
        
        Args:
            filepath: Path to PM2.5 CSV file
            validate: Whether to validate data quality
            
        Returns:
            DataFrame with processed PM2.5 data
        """
        logger.info(f"Loading PM2.5 data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} PM2.5 records")
            
            # Validate required columns
            required_cols = ['week_start', 'site_name', 'avg_pm2_5_calibrated', 
                           'site_latitude', 'site_longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert week_start to datetime
            df['week_start'] = pd.to_datetime(df['week_start'])
            
            # Create monthly aggregation
            df['year_month'] = df['week_start'].dt.to_period('M')
            
            # Handle missing values
            df['avg_pm2_5_calibrated'] = df['avg_pm2_5_calibrated'].fillna(
                df['avg_pm2_5_raw'] * 0.75 if 'avg_pm2_5_raw' in df.columns else df['avg_pm2_5_calibrated'].mean()
            )
            
            # Map sites to counties
            county_mapping = self._create_county_mapping(df['site_name'].unique())
            df['county'] = df['site_name'].map(county_mapping)
            
            # Aggregate to monthly county level
            county_monthly = df.groupby(['year_month', 'county']).agg({
                'avg_pm2_5_calibrated': 'mean',
                'avg_pm2_5_raw': 'mean' if 'avg_pm2_5_raw' in df.columns else 'first',
                'site_latitude': 'mean',
                'site_longitude': 'mean'
            }).reset_index()
            
            if validate:
                self._validate_pm25_data(county_monthly)
            
            self.pm25_data = county_monthly
            logger.info(f"Processed PM2.5 data: {len(county_monthly)} monthly county records")
            
            return county_monthly
            
        except Exception as e:
            logger.error(f"Error loading PM2.5 data: {e}")
            raise
    
    def load_health_data(self, pneumonia_file: str, tb_file: str, 
                        validate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load and process health outcome data
        
        Args:
            pneumonia_file: Path to pneumonia CSV file
            tb_file: Path to TB CSV file
            validate: Whether to validate data quality
            
        Returns:
            Dictionary containing processed health data
        """
        logger.info("Loading health outcome data")
        
        # Load pneumonia data
        try:
            pneumonia_df = pd.read_csv(pneumonia_file)
            pneumonia_df['period'] = pd.to_datetime(pneumonia_df['periodname'])
            pneumonia_df['year_month'] = pneumonia_df['period'].dt.to_period('M')
            
            # Calculate total pneumonia cases
            pneumonia_cols = [col for col in pneumonia_df.columns if 'Pneumonia' in col]
            if pneumonia_cols:
                pneumonia_df['total_pneumonia'] = pneumonia_df[pneumonia_cols].sum(axis=1, skipna=True)
            else:
                pneumonia_df['total_pneumonia'] = 0
            
            logger.info(f"Loaded {len(pneumonia_df)} pneumonia records")
            
        except Exception as e:
            logger.error(f"Error loading pneumonia data: {e}")
            raise
        
        # Load TB data
        try:
            tb_df = pd.read_csv(tb_file)
            tb_df['period'] = pd.to_datetime(tb_df['periodname'])
            tb_df['year_month'] = tb_df['period'].dt.to_period('M')
            
            # Calculate total TB cases (prioritize diagnosed over presumptive)
            tb_diagnosed_cols = [col for col in tb_df.columns if 'diagnosed with TB' in col]
            tb_presumptive_cols = [col for col in tb_df.columns if 'presumptive TB' in col]
            
            if tb_diagnosed_cols:
                tb_df['total_tb'] = tb_df[tb_diagnosed_cols].sum(axis=1, skipna=True)
            elif tb_presumptive_cols:
                tb_df['total_tb'] = tb_df[tb_presumptive_cols].sum(axis=1, skipna=True)
            else:
                tb_df['total_tb'] = 0
            
            logger.info(f"Loaded {len(tb_df)} TB records")
            
        except Exception as e:
            logger.error(f"Error loading TB data: {e}")
            raise
        
        self.health_data = {
            'pneumonia': pneumonia_df,
            'tb': tb_df
        }
        
        if validate:
            self._validate_health_data()
        
        return self.health_data
    
    def merge_data(self, add_features: bool = True) -> pd.DataFrame:
        """
        Merge environmental and health data
        
        Args:
            add_features: Whether to add temporal and lag features
            
        Returns:
            Merged DataFrame with all data
        """
        if self.pm25_data is None or not self.health_data:
            raise ValueError("Load PM2.5 and health data first before merging")
        
        logger.info("Merging environmental and health data")
        
        # Start with PM2.5 data
        merged = self.pm25_data.copy()
        
        # Merge pneumonia data
        if 'pneumonia' in self.health_data:
            pneumonia_summary = self.health_data['pneumonia'].groupby(
                ['year_month', 'organisationunitname']
            )['total_pneumonia'].sum().reset_index()
            pneumonia_summary.columns = ['year_month', 'county', 'total_pneumonia']
            
            merged = pd.merge(
                merged, pneumonia_summary,
                on=['year_month', 'county'],
                how='left'
            )
        
        # Merge TB data
        if 'tb' in self.health_data:
            tb_summary = self.health_data['tb'].groupby(
                ['year_month', 'organisationunitname']
            )['total_tb'].sum().reset_index()
            tb_summary.columns = ['year_month', 'county', 'total_tb']
            
            merged = pd.merge(
                merged, tb_summary,
                on=['year_month', 'county'],
                how='left'
            )
        
        # Fill missing values
        merged['total_pneumonia'] = merged['total_pneumonia'].fillna(0)
        merged['total_tb'] = merged['total_tb'].fillna(0)
        
        if add_features:
            merged = self._add_temporal_features(merged)
            merged = self._add_lag_features(merged)
        
        self.processed_data = merged
        logger.info(f"Merged data contains {len(merged)} records with {merged.shape[1]} features")
        
        return merged
    
    def _create_county_mapping(self, site_names: List[str]) -> Dict[str, str]:
        """Map monitoring sites to counties based on site names"""
        county_map = {}
        
        # Define patterns for county identification
        county_patterns = {
            'Kampala': ['Kampala', 'kampala', 'Central', 'Civic'],
            'Nakawa': ['Nakawa', 'nakawa', 'Butabika', 'Bugolobi', 'Ntinda', 'Mbuya', 'Luzira'],
            'Makindye': ['Makindye', 'makindye', 'Kisugu', 'Luwafu', 'Lukuli', 'Nsambya', 'Ggaba'],
            'Kawempe': ['Kawempe', 'kawempe', 'Makerere', 'Bukoto', 'Kalerwe', 'Kyebando', 'Mpererwe'],
            'Rubaga': ['Rubaga', 'rubaga', 'Kasubi', 'Busega'],
            'Wakiso': ['Wakiso', 'wakiso', 'Nansana'],
            'Kira': ['Kira', 'kira', 'Bukasa', 'Nsawo', 'Municipal']
        }
        
        for site in site_names:
            if pd.isna(site):
                continue
            
            site_str = str(site)
            mapped = False
            
            for county, patterns in county_patterns.items():
                for pattern in patterns:
                    if pattern in site_str:
                        county_map[site] = county
                        mapped = True
                        break
                if mapped:
                    break
            
            if not mapped:
                county_map[site] = 'Other'
        
        return county_map
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the dataframe"""
        df = df.copy()
        df['month'] = df['year_month'].dt.month
        df['year'] = df['year_month'].dt.year
        df['quarter'] = df['year_month'].dt.quarter
        df['season'] = df['month'].apply(lambda x: 'dry' if x in [12, 1, 2, 6, 7, 8] else 'wet')
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """Add lag features for time series analysis"""
        df = df.copy()
        
        for lag in lags:
            df[f'pm25_lag_{lag}'] = df.groupby('county')['avg_pm2_5_calibrated'].shift(lag)
            df[f'pneumonia_lag_{lag}'] = df.groupby('county')['total_pneumonia'].shift(lag)
            df[f'tb_lag_{lag}'] = df.groupby('county')['total_tb'].shift(lag)
        
        return df
    
    def _validate_pm25_data(self, df: pd.DataFrame):
        """Validate PM2.5 data quality"""
        # Check for negative values
        if (df['avg_pm2_5_calibrated'] < 0).any():
            logger.warning("Negative PM2.5 values detected, setting to 0")
            df.loc[df['avg_pm2_5_calibrated'] < 0, 'avg_pm2_5_calibrated'] = 0
        
        # Check for extreme values
        upper_limit = df['avg_pm2_5_calibrated'].quantile(0.99) * 3
        if (df['avg_pm2_5_calibrated'] > upper_limit).any():
            logger.warning(f"Extreme PM2.5 values (>{upper_limit:.1f}) detected")
        
        # Check for data gaps
        date_range = pd.date_range(
            df['year_month'].min().to_timestamp(),
            df['year_month'].max().to_timestamp(),
            freq='M'
        )
        missing_months = len(date_range) - df['year_month'].nunique()
        if missing_months > 0:
            logger.warning(f"{missing_months} months missing from PM2.5 data")
    
    def _validate_health_data(self):
        """Validate health data quality"""
        for disease, df in self.health_data.items():
            # Check for negative values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if (df[numeric_cols] < 0).any().any():
                logger.warning(f"Negative values detected in {disease} data")
            
            # Check for missing counties
            if df['organisationunitname'].isna().any():
                logger.warning(f"Missing county names in {disease} data")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of loaded data
        
        Returns:
            Dictionary with data summary
        """
        summary = {
            'pm25_data': None,
            'health_data': {},
            'merged_data': None
        }
        
        if self.pm25_data is not None:
            summary['pm25_data'] = {
                'records': len(self.pm25_data),
                'counties': self.pm25_data['county'].nunique(),
                'date_range': f"{self.pm25_data['year_month'].min()} to {self.pm25_data['year_month'].max()}",
                'mean_pm25': self.pm25_data['avg_pm2_5_calibrated'].mean(),
                'missing_values': self.pm25_data.isnull().sum().to_dict()
            }
        
        for disease, df in self.health_data.items():
            summary['health_data'][disease] = {
                'records': len(df),
                'counties': df['organisationunitname'].nunique() if 'organisationunitname' in df.columns else 0,
                'total_cases': df[f'total_{disease}'].sum() if f'total_{disease}' in df.columns else 0
            }
        
        if self.processed_data is not None:
            summary['merged_data'] = {
                'records': len(self.processed_data),
                'features': self.processed_data.shape[1],
                'counties': self.processed_data['county'].nunique(),
                'date_range': f"{self.processed_data['year_month'].min()} to {self.processed_data['year_month'].max()}"
            }
        
        return summary
    
    def save_processed_data(self, output_dir: str = './processed_data'):
        """
        Save processed data to CSV files
        
        Args:
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.processed_data is not None:
            filepath = output_path / 'merged_data.csv'
            self.processed_data.to_csv(filepath, index=False)
            logger.info(f"Saved merged data to {filepath}")
        
        if self.pm25_data is not None:
            filepath = output_path / 'pm25_monthly.csv'
            self.pm25_data.to_csv(filepath, index=False)
            logger.info(f"Saved PM2.5 data to {filepath}")
        
        for disease, df in self.health_data.items():
            filepath = output_path / f'{disease}_processed.csv'
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {disease} data to {filepath}")