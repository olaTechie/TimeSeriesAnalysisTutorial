"""
Spatial Analysis Module for Geographic Pattern Detection
This module analyzes spatial patterns and differential impacts across counties.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SpatialAnalyzer:
    """Analyze spatial patterns and differential impacts"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize spatial analyzer
        
        Args:
            data: DataFrame with spatial and health data
        """
        self.data = data.copy()
        self.county_stats = None
        self.hotspots = None
        self.spatial_correlations = None
        logger.info(f"SpatialAnalyzer initialized with {len(data)} records from {data['county'].nunique()} counties")
    
    def analyze_county_variations(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze variations across counties
        
        Args:
            metrics: List of metrics to analyze (None for default)
            
        Returns:
            DataFrame with county-level statistics
        """
        if metrics is None:
            metrics = ['avg_pm2_5_calibrated', 'total_pneumonia', 'total_tb']
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in self.data.columns]
        
        # Calculate statistics for each county
        county_stats = self.data.groupby('county').agg({
            metric: ['mean', 'std', 'min', 'max', 'median']
            for metric in available_metrics
        })
        
        # Flatten column names
        county_stats.columns = ['_'.join(col).strip() for col in county_stats.columns.values]
        
        # Add count of observations
        county_stats['n_observations'] = self.data.groupby('county').size()
        
        # Calculate coefficient of variation for each metric
        for metric in available_metrics:
            if f'{metric}_mean' in county_stats.columns and f'{metric}_std' in county_stats.columns:
                county_stats[f'{metric}_cv'] = (
                    county_stats[f'{metric}_std'] / 
                    county_stats[f'{metric}_mean'].replace(0, np.nan)
                )
        
        # Calculate vulnerability index
        county_stats = self._calculate_vulnerability_index(county_stats, available_metrics)
        
        # Add geographic coordinates if available
        if 'site_latitude' in self.data.columns and 'site_longitude' in self.data.columns:
            geo_stats = self.data.groupby('county')[['site_latitude', 'site_longitude']].mean()
            county_stats = county_stats.join(geo_stats)
        
        self.county_stats = county_stats
        logger.info(f"Analyzed {len(county_stats)} counties across {len(available_metrics)} metrics")
        
        return county_stats
    
    def identify_hotspots(self, threshold_percentile: float = 75,
                         criteria: Optional[Dict] = None) -> pd.DataFrame:
        """
        Identify pollution and disease hotspots
        
        Args:
            threshold_percentile: Percentile threshold for hotspot identification
            criteria: Custom criteria for hotspot identification
            
        Returns:
            DataFrame with identified hotspots
        """
        if criteria is None:
            criteria = {
                'pm25': 'avg_pm2_5_calibrated',
                'health': ['total_pneumonia', 'total_tb']
            }
        
        hotspots_list = []
        
        # Calculate thresholds
        thresholds = {}
        
        # PM2.5 threshold
        if 'pm25' in criteria and criteria['pm25'] in self.data.columns:
            thresholds['pm25'] = self.data[criteria['pm25']].quantile(threshold_percentile / 100)
        
        # Health thresholds
        if 'health' in criteria:
            health_metrics = criteria['health']
            if isinstance(health_metrics, str):
                health_metrics = [health_metrics]
            
            for metric in health_metrics:
                if metric in self.data.columns:
                    thresholds[metric] = self.data[metric].quantile(threshold_percentile / 100)
        
        # Identify hotspots for each time period and county
        for (year_month, county), group in self.data.groupby(['year_month', 'county']):
            hotspot_types = []
            
            # Check PM2.5
            if 'pm25' in thresholds:
                pm25_value = group[criteria['pm25']].mean()
                if pm25_value > thresholds['pm25']:
                    hotspot_types.append('PM2.5')
            
            # Check health metrics
            for metric, threshold in thresholds.items():
                if metric != 'pm25' and metric in group.columns:
                    if group[metric].mean() > threshold:
                        hotspot_types.append(metric)
            
            if hotspot_types:
                hotspot_record = {
                    'year_month': year_month,
                    'county': county,
                    'hotspot_type': ', '.join(hotspot_types),
                    'n_criteria_met': len(hotspot_types)
                }
                
                # Add values for each metric
                if 'pm25' in criteria and criteria['pm25'] in group.columns:
                    hotspot_record['pm25_value'] = group[criteria['pm25']].mean()
                
                for metric in health_metrics if 'health' in criteria else []:
                    if metric in group.columns:
                        hotspot_record[f'{metric}_value'] = group[metric].mean()
                
                hotspots_list.append(hotspot_record)
        
        self.hotspots = pd.DataFrame(hotspots_list)
        
        if len(self.hotspots) > 0:
            # Add severity score
            self.hotspots['severity_score'] = self._calculate_severity_score(self.hotspots)
            
            # Sort by severity
            self.hotspots = self.hotspots.sort_values('severity_score', ascending=False)
        
        logger.info(f"Identified {len(self.hotspots)} hotspot instances")
        
        return self.hotspots
    
    def calculate_spatial_correlations(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate spatial correlations between variables
        
        Args:
            variables: List of variables to correlate (None for default)
            
        Returns:
            DataFrame with correlation matrix
        """
        if variables is None:
            variables = ['avg_pm2_5_calibrated', 'total_pneumonia', 'total_tb']
        
        # Filter to available variables
        available_vars = [v for v in variables if v in self.data.columns]
        
        if len(available_vars) < 2:
            logger.warning("Not enough variables for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlations by county
        correlations = {}
        
        for county in self.data['county'].unique():
            county_data = self.data[self.data['county'] == county][available_vars]
            
            if len(county_data) > 3:  # Need at least 4 observations
                corr_matrix = county_data.corr()
                correlations[county] = corr_matrix
        
        # Calculate overall correlations
        overall_corr = self.data[available_vars].corr()
        
        self.spatial_correlations = {
            'overall': overall_corr,
            'by_county': correlations
        }
        
        logger.info(f"Calculated spatial correlations for {len(correlations)} counties")
        
        return overall_corr
    
    def analyze_intervention_equity(self, intervention_results: Dict) -> pd.DataFrame:
        """
        Analyze equity impacts of interventions across counties
        
        Args:
            intervention_results: Results from intervention simulations
            
        Returns:
            DataFrame with equity analysis
        """
        equity_analysis = []
        
        if self.county_stats is None:
            self.analyze_county_variations()
        
        for scenario_name, results in intervention_results.items():
            if 'health_impacts' not in results:
                continue
            
            health_data = results['health_impacts']
            
            for county in health_data['county'].unique():
                county_data = health_data[health_data['county'] == county]
                
                # Calculate benefits
                benefits = {}
                for disease in ['pneumonia', 'tb']:
                    if f'{disease}_cases_prevented' in county_data.columns:
                        benefits[f'{disease}_prevented'] = county_data[f'{disease}_cases_prevented'].sum()
                
                # Get baseline vulnerability
                if county in self.county_stats.index:
                    vulnerability = self.county_stats.loc[county, 'vulnerability_index']
                else:
                    vulnerability = 0.5  # Default middle value
                
                equity_analysis.append({
                    'scenario': scenario_name,
                    'county': county,
                    'vulnerability_index': vulnerability,
                    'total_cases_prevented': sum(benefits.values()),
                    **benefits
                })
        
        equity_df = pd.DataFrame(equity_analysis)
        
        if len(equity_df) > 0:
            # Calculate equity metrics
            for scenario in equity_df['scenario'].unique():
                scenario_data = equity_df[equity_df['scenario'] == scenario]
                
                # Correlation between vulnerability and benefits
                if len(scenario_data) > 2:
                    corr, p_value = stats.pearsonr(
                        scenario_data['vulnerability_index'],
                        scenario_data['total_cases_prevented']
                    )
                    
                    logger.info(f"Scenario {scenario}: Vulnerability-Benefit correlation = {corr:.3f} (p={p_value:.3f})")
        
        return equity_df
    
    def calculate_distance_effects(self, reference_points: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate distance-based effects from reference points
        
        Args:
            reference_points: DataFrame with reference locations (e.g., pollution sources)
            
        Returns:
            DataFrame with distance effects
        """
        if 'site_latitude' not in self.data.columns or 'site_longitude' not in self.data.columns:
            logger.warning("Geographic coordinates not available")
            return pd.DataFrame()
        
        # If no reference points provided, use highest PM2.5 locations
        if reference_points is None:
            top_pollution = self.data.nlargest(5, 'avg_pm2_5_calibrated')[
                ['site_latitude', 'site_longitude', 'avg_pm2_5_calibrated']
            ]
            reference_points = top_pollution
        
        # Calculate distances for each county
        distance_effects = []
        
        for county in self.data['county'].unique():
            county_data = self.data[self.data['county'] == county]
            
            if len(county_data) == 0:
                continue
            
            # Get county centroid
            county_lat = county_data['site_latitude'].mean()
            county_lon = county_data['site_longitude'].mean()
            
            # Calculate distances to reference points
            distances = []
            for _, ref in reference_points.iterrows():
                # Simple Euclidean distance (for small areas)
                # For larger areas, use haversine formula
                dist = np.sqrt(
                    (county_lat - ref['site_latitude'])**2 + 
                    (county_lon - ref['site_longitude'])**2
                )
                distances.append(dist)
            
            min_distance = min(distances) if distances else 0
            mean_distance = np.mean(distances) if distances else 0
            
            # Get health metrics
            health_burden = county_data[['total_pneumonia', 'total_tb']].sum().sum()
            
            distance_effects.append({
                'county': county,
                'latitude': county_lat,
                'longitude': county_lon,
                'min_distance_to_source': min_distance,
                'mean_distance_to_sources': mean_distance,
                'total_health_burden': health_burden,
                'avg_pm25': county_data['avg_pm2_5_calibrated'].mean()
            })
        
        return pd.DataFrame(distance_effects)
    
    def generate_spatial_summary(self) -> Dict:
        """
        Generate comprehensive spatial analysis summary
        
        Returns:
            Dictionary with spatial analysis summary
        """
        summary = {
            'n_counties': self.data['county'].nunique(),
            'date_range': f"{self.data['year_month'].min()} to {self.data['year_month'].max()}",
            'total_observations': len(self.data)
        }
        
        # County statistics
        if self.county_stats is not None:
            summary['highest_pm25_county'] = self.county_stats['avg_pm2_5_calibrated_mean'].idxmax()
            summary['highest_vulnerability_county'] = self.county_stats['vulnerability_index'].idxmax()
            summary['pm25_spatial_cv'] = self.county_stats['avg_pm2_5_calibrated_mean'].std() / \
                                        self.county_stats['avg_pm2_5_calibrated_mean'].mean()
        
        # Hotspot statistics
        if self.hotspots is not None and len(self.hotspots) > 0:
            summary['n_hotspot_instances'] = len(self.hotspots)
            summary['counties_with_hotspots'] = self.hotspots['county'].nunique()
            summary['most_frequent_hotspot_type'] = self.hotspots['hotspot_type'].mode().iloc[0] \
                                                    if len(self.hotspots) > 0 else None
        
        # Correlation statistics
        if self.spatial_correlations is not None:
            overall_corr = self.spatial_correlations['overall']
            if 'avg_pm2_5_calibrated' in overall_corr.index and 'total_pneumonia' in overall_corr.columns:
                summary['pm25_pneumonia_correlation'] = overall_corr.loc['avg_pm2_5_calibrated', 'total_pneumonia']
        
        return summary
    
    def _calculate_vulnerability_index(self, county_stats: pd.DataFrame, 
                                      metrics: List[str]) -> pd.DataFrame:
        """Calculate vulnerability index for counties"""
        county_stats = county_stats.copy()
        
        # Components of vulnerability index
        components = []
        weights = []
        
        # PM2.5 exposure (40% weight)
        if 'avg_pm2_5_calibrated_mean' in county_stats.columns:
            pm25_normalized = county_stats['avg_pm2_5_calibrated_mean'] / county_stats['avg_pm2_5_calibrated_mean'].max()
            components.append(pm25_normalized)
            weights.append(0.4)
        
        # Health burden (40% weight)
        health_burden = 0
        health_count = 0
        for disease in ['pneumonia', 'tb']:
            col_name = f'total_{disease}_mean'
            if col_name in county_stats.columns:
                health_burden += county_stats[col_name] / county_stats[col_name].max()
                health_count += 1
        
        if health_count > 0:
            components.append(health_burden / health_count)
            weights.append(0.4)
        
        # Variability (20% weight) - higher variability = higher vulnerability
        variability = 0
        var_count = 0
        for metric in metrics:
            cv_col = f'{metric}_cv'
            if cv_col in county_stats.columns:
                variability += county_stats[cv_col].fillna(0) / county_stats[cv_col].max()
                var_count += 1
        
        if var_count > 0:
            components.append(variability / var_count)
            weights.append(0.2)
        
        # Calculate weighted index
        if components:
            weights = np.array(weights) / sum(weights)  # Normalize weights
            vulnerability_index = sum(w * c for w, c in zip(weights, components))
            county_stats['vulnerability_index'] = vulnerability_index
        else:
            county_stats['vulnerability_index'] = 0.5  # Default middle value
        
        return county_stats
    
    def _calculate_severity_score(self, hotspots: pd.DataFrame) -> np.ndarray:
        """Calculate severity score for hotspots"""
        severity_scores = np.zeros(len(hotspots))
        
        # Normalize values for scoring
        for col in hotspots.columns:
            if col.endswith('_value'):
                if hotspots[col].max() > 0:
                    normalized = hotspots[col] / hotspots[col].max()
                    severity_scores += normalized.values
        
        # Add weight for multiple criteria
        severity_scores += hotspots['n_criteria_met'].values * 0.2
        
        # Normalize to 0-1 scale
        if severity_scores.max() > 0:
            severity_scores = severity_scores / severity_scores.max()
        
        return severity_scores
    
    def export_spatial_results(self, output_dir: str = './spatial_results'):
        """
        Export spatial analysis results to CSV files
        
        Args:
            output_dir: Directory to save results
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export county statistics
        if self.county_stats is not None:
            self.county_stats.to_csv(output_path / 'county_statistics.csv')
            logger.info(f"Saved county statistics to {output_path / 'county_statistics.csv'}")
        
        # Export hotspots
        if self.hotspots is not None and len(self.hotspots) > 0:
            self.hotspots.to_csv(output_path / 'identified_hotspots.csv', index=False)
            logger.info(f"Saved hotspots to {output_path / 'identified_hotspots.csv'}")
        
        # Export correlations
        if self.spatial_correlations is not None:
            self.spatial_correlations['overall'].to_csv(output_path / 'spatial_correlations.csv')
            logger.info(f"Saved correlations to {output_path / 'spatial_correlations.csv'}")