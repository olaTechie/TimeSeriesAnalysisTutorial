"""
Visualization Module for Air Quality Analysis
This module creates comprehensive visualizations for the analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Visualizer:
    """Create comprehensive visualizations for analysis results"""
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize visualizer
        
        Args:
            figure_size: Default figure size
            dpi: Dots per inch for figures
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.figures = {}
        logger.info("Visualizer initialized")
    
    def plot_time_series(self, data: pd.DataFrame, 
                        variables: List[str],
                        title: str = "Time Series Analysis",
                        by_county: bool = False) -> plt.Figure:
        """
        Plot time series data
        
        Args:
            data: DataFrame with time series data
            variables: List of variables to plot
            title: Plot title
            by_county: Whether to separate by county
            
        Returns:
            Matplotlib figure
        """
        if by_county and 'county' in data.columns:
            counties = data['county'].unique()
            n_counties = len(counties)
            n_cols = min(3, n_counties)
            n_rows = (n_counties + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figure_size[0]*1.5, self.figure_size[1]))
            axes = axes.flatten() if n_counties > 1 else [axes]
            
            for idx, county in enumerate(counties):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                county_data = data[data['county'] == county].sort_values('year_month')
                
                for var in variables:
                    if var in county_data.columns:
                        ax.plot(county_data['year_month'].astype(str), 
                               county_data[var], 
                               label=var, marker='o', markersize=3)
                
                ax.set_title(f'{county}')
                ax.set_xlabel('Time')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(counties), len(axes)):
                axes[idx].set_visible(False)
            
        else:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Aggregate if multiple counties
            if 'county' in data.columns:
                plot_data = data.groupby('year_month')[variables].mean()
            else:
                plot_data = data.set_index('year_month')[variables]
            
            for var in variables:
                if var in plot_data.columns:
                    ax.plot(plot_data.index.astype(str), plot_data[var], 
                           label=var, marker='o', markersize=4)
            
            ax.set_xlabel('Time Period', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['time_series'] = fig
        return fig
    
    def plot_scenario_comparison(self, simulation_results: Dict,
                                metrics: Optional[List[str]] = None) -> plt.Figure:
        """
        Compare PM2.5 levels across scenarios
        
        Args:
            simulation_results: Dictionary with simulation results
            metrics: Metrics to compare (None for default)
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['simulated_pm25', 'pm25_reduction_pct']
        
        n_scenarios = len(simulation_results)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, min(3, n_scenarios), 
                                figsize=(self.figure_size[0]*1.5, self.figure_size[1]))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        if n_scenarios == 1:
            axes = axes.reshape(-1, 1)
        
        for col_idx, (scenario_name, results) in enumerate(simulation_results.items()):
            if col_idx >= 3:  # Limit to 3 columns
                break
            
            data = results.get('simulated_data', results.get('health_impacts', pd.DataFrame()))
            
            for row_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx] if n_scenarios > 1 else axes[row_idx]
                
                if metric in data.columns:
                    # Time series plot
                    if 'year_month' in data.columns:
                        grouped = data.groupby('year_month')[metric].mean()
                        ax.plot(range(len(grouped)), grouped.values, linewidth=2)
                        ax.set_xlabel('Time Period')
                    else:
                        ax.hist(data[metric], bins=20, edgecolor='black', alpha=0.7)
                        ax.set_xlabel(metric)
                    
                    ax.set_ylabel('Value')
                    ax.set_title(f"{results['scenario'].name}\n{metric}")
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle('Scenario Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        self.figures['scenario_comparison'] = fig
        return fig
    
    def plot_health_impacts(self, simulation_results: Dict) -> plt.Figure:
        """
        Visualize health impacts across scenarios
        
        Args:
            simulation_results: Dictionary with simulation results
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        scenarios = []
        pneumonia_prevented = []
        tb_prevented = []
        pm25_reduction = []
        
        for scenario_name, results in simulation_results.items():
            scenario = results['scenario']
            scenarios.append(scenario.name.replace(' ', '\n'))  # Line break for long names
            
            # Get health impacts
            stats = results.get('summary_stats', {})
            pneumonia_prevented.append(stats.get('pneumonia_total_prevented', 0))
            tb_prevented.append(stats.get('tb_total_prevented', 0))
            pm25_reduction.append(scenario.pm25_reduction * 100)
        
        # Plot 1: Cases prevented
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pneumonia_prevented, width, label='Pneumonia', color='skyblue')
        bars2 = ax1.bar(x + width/2, tb_prevented, width, label='TB', color='lightcoral')
        
        ax1.set_xlabel('Intervention Scenario', fontsize=12)
        ax1.set_ylabel('Cases Prevented', fontsize=12)
        ax1.set_title('Health Impact: Cases Prevented', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, fontsize=9)
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.annotate(f'{height:.0f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
        
        # Plot 2: PM2.5 reduction vs total health benefit
        total_prevented = [p + t for p, t in zip(pneumonia_prevented, tb_prevented)]
        
        ax2.scatter(pm25_reduction, total_prevented, s=100, alpha=0.6)
        
        # Add labels for each point
        for i, txt in enumerate([s.split('\n')[0] for s in scenarios]):
            ax2.annotate(txt, (pm25_reduction[i], total_prevented[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('PM2.5 Reduction (%)', fontsize=12)
        ax2.set_ylabel('Total Cases Prevented', fontsize=12)
        ax2.set_title('Intervention Effectiveness', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Health Impact Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['health_impacts'] = fig
        return fig
    
    def plot_spatial_analysis(self, spatial_stats: pd.DataFrame,
                             metric: str = 'vulnerability_index') -> plt.Figure:
        """
        Create spatial visualization of county-level impacts
        
        Args:
            spatial_stats: DataFrame with county statistics
            metric: Metric to visualize
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        # Plot 1: PM2.5 by county
        ax1 = axes[0, 0]
        if 'avg_pm2_5_calibrated_mean' in spatial_stats.columns:
            spatial_stats['avg_pm2_5_calibrated_mean'].plot(kind='bar', ax=ax1, color='coral')
            ax1.set_title('Average PM2.5 by County', fontweight='bold')
            ax1.set_ylabel('PM2.5 (μg/m³)')
            ax1.set_xlabel('County')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, axis='y', alpha=0.3)
        
        # Plot 2: Health burden by county
        ax2 = axes[0, 1]
        health_cols = [col for col in spatial_stats.columns if 'pneumonia_mean' in col or 'tb_mean' in col]
        if health_cols:
            spatial_stats[health_cols].plot(kind='bar', ax=ax2)
            ax2.set_title('Average Health Burden by County', fontweight='bold')
            ax2.set_ylabel('Cases')
            ax2.set_xlabel('County')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(['Pneumonia', 'TB'], loc='best')
            ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Vulnerability index
        ax3 = axes[1, 0]
        if metric in spatial_stats.columns:
            colors = plt.cm.RdYlGn_r(spatial_stats[metric] / spatial_stats[metric].max())
            bars = ax3.bar(range(len(spatial_stats)), spatial_stats[metric], color=colors)
            ax3.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax3.set_ylabel('Score')
            ax3.set_xlabel('County')
            ax3.set_xticks(range(len(spatial_stats)))
            ax3.set_xticklabels(spatial_stats.index, rotation=45)
            ax3.grid(True, axis='y', alpha=0.3)
        
        # Plot 4: Correlation heatmap
        ax4 = axes[1, 1]
        numeric_cols = spatial_stats.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = spatial_stats[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=ax4, cmap='coolwarm', center=0)
            ax4.set_title('Variable Correlations', fontweight='bold')
        
        plt.suptitle('Spatial Analysis Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['spatial_analysis'] = fig
        return fig
    
    def plot_time_series_decomposition(self, data: pd.DataFrame,
                                      column: str = 'avg_pm2_5_calibrated') -> plt.Figure:
        """
        Plot time series decomposition
        
        Args:
            data: DataFrame with time series data
            column: Column to decompose
            
        Returns:
            Matplotlib figure
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        fig, axes = plt.subplots(4, 1, figsize=(self.figure_size[0], self.figure_size[1]*1.2))
        
        # Prepare time series
        if 'year_month' in data.columns:
            ts_data = data.groupby('year_month')[column].mean()
        else:
            ts_data = data[column]
        
        # Check if we have enough data
        if len(ts_data) >= 24:  # Need at least 2 years for seasonal decomposition
            try:
                decomposition = seasonal_decompose(ts_data, model='additive', period=12)
                
                # Plot components
                ts_data.plot(ax=axes[0], title='Original Time Series')
                axes[0].set_ylabel(column.replace('_', ' ').title())
                
                decomposition.trend.plot(ax=axes[1], title='Trend Component')
                axes[1].set_ylabel('Trend')
                
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
                axes[2].set_ylabel('Seasonal')
                
                decomposition.resid.plot(ax=axes[3], title='Residual Component')
                axes[3].set_ylabel('Residual')
                
            except Exception as e:
                logger.warning(f"Decomposition failed: {e}")
                ts_data.plot(ax=axes[0], title='Time Series (Decomposition Failed)')
                for ax in axes[1:]:
                    ax.text(0.5, 0.5, 'Insufficient data for decomposition',
                           transform=ax.transAxes, ha='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
        else:
            ts_data.plot(ax=axes[0], title='Time Series (Insufficient data for decomposition)')
            for ax in axes[1:]:
                ax.text(0.5, 0.5, 'Need at least 24 months for decomposition',
                       transform=ax.transAxes, ha='center')
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle(f'Time Series Decomposition: {column}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['decomposition'] = fig
        return fig
    
    def plot_model_performance(self, model_results: Dict) -> plt.Figure:
        """
        Plot model performance comparison
        
        Args:
            model_results: Dictionary with model performance metrics
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        models = list(model_results.keys())
        
        # Prepare metrics
        metrics_data = {
            'Accuracy': [],
            'F1 Score': [],
            'Precision': [],
            'Recall': []
        }
        
        for model in models:
            results = model_results[model]
            metrics_data['Accuracy'].append(results.get('avg_accuracy', 0))
            metrics_data['F1 Score'].append(results.get('avg_f1', 0))
            metrics_data['Precision'].append(results.get('avg_precision', 0))
            metrics_data['Recall'].append(results.get('avg_recall', 0))
        
        # Plot each metric
        for idx, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[idx // 2, idx % 2]
            
            bars = ax.bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['model_performance'] = fig
        return fig
    
    def plot_hotspot_map(self, hotspots: pd.DataFrame) -> plt.Figure:
        """
        Plot hotspot locations and severity
        
        Args:
            hotspots: DataFrame with hotspot information
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        if len(hotspots) == 0:
            ax1.text(0.5, 0.5, 'No hotspots identified', 
                    transform=ax1.transAxes, ha='center', fontsize=12)
            ax2.text(0.5, 0.5, 'No hotspots identified', 
                    transform=ax2.transAxes, ha='center', fontsize=12)
        else:
            # Plot 1: Hotspot frequency by county
            county_counts = hotspots['county'].value_counts()
            county_counts.plot(kind='bar', ax=ax1, color='darkred')
            ax1.set_title('Hotspot Frequency by County', fontweight='bold')
            ax1.set_xlabel('County')
            ax1.set_ylabel('Number of Hotspot Instances')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Plot 2: Hotspot types distribution
            type_counts = hotspots['hotspot_type'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
            ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Distribution of Hotspot Types', fontweight='bold')
        
        plt.suptitle('Hotspot Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['hotspot_map'] = fig
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame,
                               variables: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap
        
        Args:
            data: DataFrame with variables to correlate
            variables: List of variables to include (None for all numeric)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select variables
        if variables is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            variables = [col for col in numeric_cols if not col.endswith('_lag')][:10]  # Limit to 10
        
        # Calculate correlation matrix
        corr_matrix = data[variables].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['correlation_matrix'] = fig
        return fig
    
    def create_dashboard(self, data: pd.DataFrame,
                        simulation_results: Optional[Dict] = None,
                        spatial_stats: Optional[pd.DataFrame] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            data: Main dataset
            simulation_results: Simulation results dictionary
            spatial_stats: Spatial analysis results
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # PM2.5 time series
        ax1 = fig.add_subplot(gs[0, :])
        if 'avg_pm2_5_calibrated' in data.columns and 'year_month' in data.columns:
            ts_data = data.groupby('year_month')['avg_pm2_5_calibrated'].mean()
            ax1.plot(ts_data.index.astype(str), ts_data.values, linewidth=2, color='darkblue')
            ax1.fill_between(range(len(ts_data)), ts_data.values, alpha=0.3)
            ax1.set_title('PM2.5 Levels Over Time', fontweight='bold')
            ax1.set_xlabel('Time Period')
            ax1.set_ylabel('PM2.5 (μg/m³)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # County comparison
        ax2 = fig.add_subplot(gs[1, 0])
        if spatial_stats is not None and 'avg_pm2_5_calibrated_mean' in spatial_stats.columns:
            spatial_stats['avg_pm2_5_calibrated_mean'].plot(kind='barh', ax=ax2, color='coral')
            ax2.set_title('PM2.5 by County', fontweight='bold')
            ax2.set_xlabel('PM2.5 (μg/m³)')
        
        # Health burden
        ax3 = fig.add_subplot(gs[1, 1])
        if 'total_pneumonia' in data.columns and 'total_tb' in data.columns:
            health_data = data[['total_pneumonia', 'total_tb']].sum()
            ax3.pie(health_data.values, labels=['Pneumonia', 'TB'],
                   autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
            ax3.set_title('Disease Distribution', fontweight='bold')
        
        # Scenario comparison
        ax4 = fig.add_subplot(gs[1, 2])
        if simulation_results:
            scenarios = []
            reductions = []
            for name, results in simulation_results.items():
                scenarios.append(name.replace('_', ' ').title())
                reductions.append(results['scenario'].pm25_reduction * 100)
            
            ax4.barh(scenarios, reductions, color='green', alpha=0.7)
            ax4.set_title('PM2.5 Reduction Potential', fontweight='bold')
            ax4.set_xlabel('Reduction (%)')
        
        # Correlation heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        if len(data.select_dtypes(include=[np.number]).columns) > 1:
            key_vars = ['avg_pm2_5_calibrated', 'total_pneumonia', 'total_tb']
            key_vars = [v for v in key_vars if v in data.columns]
            if len(key_vars) > 1:
                corr = data[key_vars].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', ax=ax5,
                           cmap='coolwarm', center=0)
                ax5.set_title('Key Variable Correlations', fontweight='bold')
        
        # Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = "Summary Statistics\n" + "="*20 + "\n"
        if 'county' in data.columns:
            summary_text += f"Counties: {data['county'].nunique()}\n"
        if 'year_month' in data.columns:
            summary_text += f"Time Period: {data['year_month'].min()} to {data['year_month'].max()}\n"
        if 'avg_pm2_5_calibrated' in data.columns:
            summary_text += f"Avg PM2.5: {data['avg_pm2_5_calibrated'].mean():.1f} μg/m³\n"
        if 'total_pneumonia' in data.columns:
            summary_text += f"Total Pneumonia: {data['total_pneumonia'].sum():.0f}\n"
        if 'total_tb' in data.columns:
            summary_text += f"Total TB: {data['total_tb'].sum():.0f}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Air Quality and Health Impact Dashboard', fontsize=16, fontweight='bold')
        
        self.figures['dashboard'] = fig
        return fig
    
    def save_all_figures(self, output_dir: str = './figures', format: str = 'png'):
        """
        Save all generated figures to files
        
        Args:
            output_dir: Directory to save figures
            format: File format (png, pdf, svg)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = output_path / f'{name}.{format}'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved {name} to {filepath}")
    
    def close_all_figures(self):
        """Close all matplotlib figures to free memory"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures = {}
        logger.info("Closed all figures")