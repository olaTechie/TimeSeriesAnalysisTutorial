"""
Intervention Simulation Module for Air Quality Analysis
This module simulates various air quality intervention scenarios and their health impacts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class InterventionScenario:
    """Class to define intervention scenarios"""
    name: str
    pm25_reduction: float  # Percentage reduction in PM2.5
    implementation_time: int  # Months to full implementation
    cost_factor: float  # Relative cost (1.0 = baseline)
    sustainability_score: float  # 0-1 scale
    
class InterventionSimulator:
    """Simulate various air quality intervention scenarios"""
    
    def __init__(self, baseline_data: pd.DataFrame):
        """
        Initialize intervention simulator
        
        Args:
            baseline_data: DataFrame with baseline environmental and health data
        """
        self.baseline_data = baseline_data.copy()
        self.scenarios = self._define_scenarios()
        self.simulation_results = {}
        logger.info("InterventionSimulator initialized with %d baseline records", len(baseline_data))
    
    def _define_scenarios(self) -> Dict[str, InterventionScenario]:
        """Define intervention scenarios based on literature"""
        scenarios = {
            'baseline': InterventionScenario(
                name='Baseline (No Intervention)',
                pm25_reduction=0.0,
                implementation_time=0,
                cost_factor=1.0,
                sustainability_score=0.5
            ),
            'traffic_control': InterventionScenario(
                name='Traffic Carbon Emission Controls',
                pm25_reduction=0.25,  # 25% reduction
                implementation_time=12,
                cost_factor=2.5,
                sustainability_score=0.7
            ),
            'green_spaces': InterventionScenario(
                name='Urban Green Space Expansion',
                pm25_reduction=0.15,  # 15% reduction
                implementation_time=24,
                cost_factor=3.0,
                sustainability_score=0.9
            ),
            'road_paving': InterventionScenario(
                name='Road Paving (Dust Reduction)',
                pm25_reduction=0.20,  # 20% reduction
                implementation_time=18,
                cost_factor=4.0,
                sustainability_score=0.8
            ),
            'combined_traffic_green': InterventionScenario(
                name='Combined: Traffic Control + Green Spaces',
                pm25_reduction=0.35,  # 35% reduction
                implementation_time=24,
                cost_factor=5.0,
                sustainability_score=0.85
            ),
            'all_interventions': InterventionScenario(
                name='All Interventions Combined',
                pm25_reduction=0.45,  # 45% reduction
                implementation_time=36,
                cost_factor=8.0,
                sustainability_score=0.9
            )
        }
        return scenarios
    
    def run_scenario(self, scenario_name: str, timeframe_years: int = 5,
                     include_uncertainty: bool = False) -> Dict:
        """
        Run a single intervention scenario
        
        Args:
            scenario_name: Name of the scenario to run
            timeframe_years: Number of years to simulate
            include_uncertainty: Whether to include uncertainty bounds
            
        Returns:
            Dictionary with simulation results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.scenarios.keys())}")
        
        scenario = self.scenarios[scenario_name]
        logger.info(f"Running scenario: {scenario.name}")
        
        # Simulate PM2.5 reduction
        simulated_data = self.simulate_pm25_reduction(scenario, timeframe_years)
        
        # Estimate health impacts
        health_impacts = self.estimate_health_impact(simulated_data)
        
        # Calculate uncertainty bounds if requested
        if include_uncertainty:
            uncertainty = self._calculate_uncertainty_bounds(simulated_data, scenario)
        else:
            uncertainty = None
        
        results = {
            'scenario': scenario,
            'simulated_data': simulated_data,
            'health_impacts': health_impacts,
            'summary_stats': self._calculate_summary_stats(health_impacts),
            'uncertainty': uncertainty
        }
        
        self.simulation_results[scenario_name] = results
        return results
    
    def run_all_scenarios(self, timeframe_years: int = 5,
                          include_uncertainty: bool = False) -> Dict:
        """
        Run all defined scenarios
        
        Args:
            timeframe_years: Number of years to simulate
            include_uncertainty: Whether to include uncertainty bounds
            
        Returns:
            Dictionary with all simulation results
        """
        results = {}
        
        for scenario_name in self.scenarios.keys():
            results[scenario_name] = self.run_scenario(
                scenario_name, timeframe_years, include_uncertainty
            )
        
        return results
    
    def simulate_pm25_reduction(self, scenario: InterventionScenario,
                               timeframe_years: int = 5) -> pd.DataFrame:
        """
        Simulate PM2.5 levels under intervention scenario
        
        Args:
            scenario: Intervention scenario to simulate
            timeframe_years: Number of years to project
            
        Returns:
            DataFrame with simulated PM2.5 levels
        """
        simulated_data = self.baseline_data.copy()
        
        # Calculate implementation curve (sigmoid function for gradual implementation)
        months = timeframe_years * 12
        implementation_curve = self._sigmoid_implementation(
            months, scenario.implementation_time
        )
        
        # Create future projections
        future_data = self._create_future_projections(months)
        
        # Combine historical and future data
        if future_data is not None:
            simulated_data = pd.concat([simulated_data, future_data], ignore_index=True)
        
        # Apply PM2.5 reductions
        simulated_data['implementation_factor'] = 0.0
        
        # Calculate start of intervention (assuming it starts after historical data)
        historical_months = len(self.baseline_data['year_month'].unique())
        
        for i in range(len(simulated_data)):
            month_idx = i // simulated_data['county'].nunique()
            if month_idx >= historical_months:
                future_month_idx = month_idx - historical_months
                if future_month_idx < len(implementation_curve):
                    simulated_data.loc[i, 'implementation_factor'] = implementation_curve[future_month_idx]
        
        # Apply reduction
        simulated_data['pm25_reduction_pct'] = simulated_data['implementation_factor'] * scenario.pm25_reduction
        simulated_data['simulated_pm25'] = simulated_data['avg_pm2_5_calibrated'] * (1 - simulated_data['pm25_reduction_pct'])
        
        # Add scenario metadata
        simulated_data['scenario_name'] = scenario.name
        simulated_data['scenario_cost_factor'] = scenario.cost_factor
        simulated_data['scenario_sustainability'] = scenario.sustainability_score
        
        return simulated_data
    
    def estimate_health_impact(self, simulated_data: pd.DataFrame,
                              custom_crf: Optional[Dict] = None) -> pd.DataFrame:
        """
        Estimate health impacts based on PM2.5 changes
        
        Args:
            simulated_data: DataFrame with simulated PM2.5 levels
            custom_crf: Custom concentration-response functions
            
        Returns:
            DataFrame with health impact estimates
        """
        # Default concentration-response functions (relative risk per 10 μg/m³ PM2.5)
        crf = custom_crf or {
            'pneumonia': 1.12,  # 12% increase per 10 μg/m³
            'tb': 1.08,  # 8% increase per 10 μg/m³
            'ili': 1.15,  # 15% increase per 10 μg/m³
            'sari': 1.20  # 20% increase per 10 μg/m³
        }
        
        health_impacts = simulated_data.copy()
        
        # Calculate PM2.5 changes from baseline
        baseline_pm25_by_county = self.baseline_data.groupby('county')['avg_pm2_5_calibrated'].mean()
        
        for disease, rr_per_10 in crf.items():
            # Calculate relative risk for each record
            health_impacts[f'{disease}_relative_risk'] = 1.0
            
            for county in health_impacts['county'].unique():
                if county in baseline_pm25_by_county.index:
                    baseline_pm25 = baseline_pm25_by_county[county]
                    county_mask = health_impacts['county'] == county
                    
                    # PM2.5 change from baseline
                    pm25_change = health_impacts.loc[county_mask, 'simulated_pm25'] - baseline_pm25
                    
                    # Calculate relative risk
                    relative_risk = rr_per_10 ** (pm25_change / 10)
                    health_impacts.loc[county_mask, f'{disease}_relative_risk'] = relative_risk
                    
                    # Estimate baseline cases (use historical average if available)
                    if f'total_{disease}' in self.baseline_data.columns:
                        baseline_cases = self.baseline_data[
                            self.baseline_data['county'] == county
                        ][f'total_{disease}'].mean()
                    else:
                        baseline_cases = 100  # Default placeholder
                    
                    # Calculate estimated cases and prevention
                    health_impacts.loc[county_mask, f'{disease}_baseline_cases'] = baseline_cases
                    health_impacts.loc[county_mask, f'{disease}_estimated_cases'] = baseline_cases * relative_risk
                    health_impacts.loc[county_mask, f'{disease}_cases_prevented'] = baseline_cases * (1 - relative_risk)
        
        return health_impacts
    
    def compare_scenarios(self, scenario_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple scenarios
        
        Args:
            scenario_names: List of scenarios to compare (None for all)
            
        Returns:
            DataFrame with scenario comparison
        """
        if not self.simulation_results:
            logger.warning("No simulation results available. Run scenarios first.")
            return pd.DataFrame()
        
        if scenario_names is None:
            scenario_names = list(self.simulation_results.keys())
        
        comparison_data = []
        
        for name in scenario_names:
            if name not in self.simulation_results:
                continue
            
            results = self.simulation_results[name]
            scenario = results['scenario']
            stats = results['summary_stats']
            
            comparison_data.append({
                'Scenario': scenario.name,
                'PM2.5 Reduction (%)': scenario.pm25_reduction * 100,
                'Implementation Time (months)': scenario.implementation_time,
                'Cost Factor': scenario.cost_factor,
                'Sustainability Score': scenario.sustainability_score,
                'Mean PM2.5 Final': stats.get('pm25_final_mean', 0),
                'Total Cases Prevented': stats.get('total_cases_prevented', 0),
                'Cost per Case Prevented': stats.get('cost_per_case_prevented', 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    def calculate_cost_effectiveness(self, scenario_name: str,
                                    baseline_cost: float = 1000000) -> Dict:
        """
        Calculate cost-effectiveness of intervention
        
        Args:
            scenario_name: Name of scenario to analyze
            baseline_cost: Baseline annual cost in currency units
            
        Returns:
            Dictionary with cost-effectiveness metrics
        """
        if scenario_name not in self.simulation_results:
            raise ValueError(f"Scenario {scenario_name} not found in results")
        
        results = self.simulation_results[scenario_name]
        scenario = results['scenario']
        health_impacts = results['health_impacts']
        
        # Calculate total costs
        total_cost = baseline_cost * scenario.cost_factor
        
        # Calculate total health benefits
        total_cases_prevented = sum([
            health_impacts[f'{d}_cases_prevented'].sum()
            for d in ['pneumonia', 'tb']
            if f'{d}_cases_prevented' in health_impacts.columns
        ])
        
        # Calculate cost-effectiveness ratios
        cost_effectiveness = {
            'total_cost': total_cost,
            'total_cases_prevented': total_cases_prevented,
            'cost_per_case_prevented': total_cost / max(total_cases_prevented, 1),
            'pm25_reduction_achieved': health_impacts['pm25_reduction_pct'].mean() * 100,
            'cost_per_percent_pm25_reduction': total_cost / max(scenario.pm25_reduction * 100, 0.01),
            'sustainability_adjusted_cost': total_cost / scenario.sustainability_score
        }
        
        return cost_effectiveness
    
    def _sigmoid_implementation(self, total_months: int,
                               implementation_time: int) -> np.ndarray:
        """Generate sigmoid curve for gradual implementation"""
        if implementation_time == 0:
            return np.ones(total_months)
        
        x = np.linspace(0, 10, total_months)
        center = 5 * (implementation_time / 12)  # Convert to curve scale
        sigmoid = 1 / (1 + np.exp(-(x - center)))
        return sigmoid
    
    def _create_future_projections(self, months: int) -> Optional[pd.DataFrame]:
        """Create future data projections"""
        if len(self.baseline_data) == 0:
            return None
        
        future_data = []
        counties = self.baseline_data['county'].unique()
        
        # Get the last date in baseline data
        last_date = self.baseline_data['year_month'].max()
        
        for i in range(1, months + 1):
            # Create future month
            future_month = last_date + i
            
            for county in counties:
                county_data = self.baseline_data[self.baseline_data['county'] == county]
                
                # Use seasonal average for PM2.5 projection
                month = (last_date.month + i) % 12
                if month == 0:
                    month = 12
                
                seasonal_pm25 = county_data[county_data['month'] == month]['avg_pm2_5_calibrated'].mean() \
                               if 'month' in county_data.columns \
                               else county_data['avg_pm2_5_calibrated'].mean()
                
                future_row = {
                    'county': county,
                    'year_month': future_month,
                    'avg_pm2_5_calibrated': seasonal_pm25,
                    'month': month,
                    'year': future_month.year,
                    'total_pneumonia': county_data['total_pneumonia'].mean() if 'total_pneumonia' in county_data.columns else 0,
                    'total_tb': county_data['total_tb'].mean() if 'total_tb' in county_data.columns else 0
                }
                future_data.append(future_row)
        
        return pd.DataFrame(future_data)
    
    def _calculate_summary_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate summary statistics for scenario"""
        stats = {}
        
        # PM2.5 statistics
        if 'simulated_pm25' in data.columns:
            stats['pm25_final_mean'] = data['simulated_pm25'].iloc[-1] if len(data) > 0 else 0
            stats['pm25_reduction_mean'] = data['pm25_reduction_pct'].mean() * 100
            stats['pm25_reduction_max'] = data['pm25_reduction_pct'].max() * 100
        
        # Health impact statistics
        for disease in ['pneumonia', 'tb', 'ili', 'sari']:
            if f'{disease}_cases_prevented' in data.columns:
                stats[f'{disease}_total_prevented'] = data[f'{disease}_cases_prevented'].sum()
                stats[f'{disease}_mean_prevented'] = data[f'{disease}_cases_prevented'].mean()
        
        stats['total_cases_prevented'] = sum([
            stats.get(f'{d}_total_prevented', 0)
            for d in ['pneumonia', 'tb', 'ili', 'sari']
        ])
        
        return stats
    
    def _calculate_uncertainty_bounds(self, data: pd.DataFrame,
                                     scenario: InterventionScenario,
                                     confidence: float = 0.95) -> Dict:
        """Calculate uncertainty bounds for predictions"""
        bounds = {}
        
        # PM2.5 reduction uncertainty (±20% of estimated reduction)
        reduction_std = scenario.pm25_reduction * 0.2
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        bounds['pm25_reduction'] = {
            'mean': scenario.pm25_reduction,
            'lower': max(0, scenario.pm25_reduction - z_score * reduction_std),
            'upper': min(1, scenario.pm25_reduction + z_score * reduction_std)
        }
        
        # Health impact uncertainty
        for disease in ['pneumonia', 'tb']:
            if f'{disease}_cases_prevented' in data.columns:
                prevented = data[f'{disease}_cases_prevented']
                bounds[f'{disease}_prevented'] = {
                    'mean': prevented.mean(),
                    'lower': prevented.quantile((1 - confidence) / 2),
                    'upper': prevented.quantile((1 + confidence) / 2)
                }
        
        return bounds
    
    def export_results(self, output_dir: str = './simulation_results'):
        """
        Export simulation results to CSV files
        
        Args:
            output_dir: Directory to save results
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export comparison table
        if self.simulation_results:
            comparison = self.compare_scenarios()
            comparison.to_csv(output_path / 'scenario_comparison.csv', index=False)
            logger.info(f"Saved scenario comparison to {output_path / 'scenario_comparison.csv'}")
        
        # Export individual scenario results
        for scenario_name, results in self.simulation_results.items():
            # Save simulated data
            results['simulated_data'].to_csv(
                output_path / f'{scenario_name}_simulated.csv',
                index=False
            )
            
            # Save health impacts
            results['health_impacts'].to_csv(
                output_path / f'{scenario_name}_health_impacts.csv',
                index=False
            )
            
            logger.info(f"Saved {scenario_name} results to {output_path}")