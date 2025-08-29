# timeseries_health_analysis.py
"""
Time Series Analysis for Public Health Module
A comprehensive toolkit for analyzing air quality and health data relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Statistical imports
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import partial_dependence

# Scipy imports
from scipy import stats
from scipy.stats import jarque_bera

warnings.filterwarnings('ignore')


@dataclass
class ITSAResults:
    """Container for ITSA analysis results"""
    baseline_level: float
    pre_trend: float
    level_change: float
    trend_change: float
    model: Any
    predictions: pd.Series
    counterfactual: pd.Series
    p_values: Dict[str, float]


@dataclass
class SimulationResults:
    """Container for intervention simulation results"""
    scenario_name: str
    pm25_reduction: float
    new_pm25: float
    cases_prevented_monthly: float
    cases_prevented_total: float
    relative_risk: float
    cost_per_case: float
    benefit_cost_ratio: float
    net_present_value: float


class TimeSeriesHealthAnalyzer:
    """
    Main class for time series analysis of health and environmental data
    """
    
    def __init__(self, data: pd.DataFrame, date_col: str = 'ds', 
                 pm_col: str = 'pm2_5', health_col: str = 'TB'):
        """
        Initialize the analyzer with data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with time series
        date_col : str
            Name of date column
        pm_col : str
            Name of PM2.5 or pollution column
        health_col : str
            Name of health outcome column
        """
        self.data = data.copy()
        self.date_col = date_col
        self.pm_col = pm_col
        self.health_col = health_col
        
        # Convert date and set index
        if date_col in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data[date_col])
            self.data = self.data.set_index('date').sort_index()
        
        # Store original data
        self.original_data = self.data.copy()
        
        # Results storage
        self.itsa_results = None
        self.seasonal_decomposition = None
        self.ml_model = None
        self.simulation_results = {}
        
    def run_itsa(self, intervention_date: str) -> ITSAResults:
        """
        Perform Interrupted Time Series Analysis
        
        Parameters:
        -----------
        intervention_date : str
            Date of intervention in 'YYYY-MM-DD' format
        
        Returns:
        --------
        ITSAResults object containing analysis results
        """
        intervention_date = pd.to_datetime(intervention_date)
        
        # Create ITSA variables
        self.data['time'] = range(len(self.data))
        self.data['intervention'] = (self.data.index >= intervention_date).astype(int)
        
        # Calculate time since intervention
        intervention_start = self.data[self.data['intervention'] == 1]['time'].iloc[0] if any(self.data['intervention'] == 1) else len(self.data)
        self.data['time_since_intervention'] = np.where(
            self.data['intervention'] == 1,
            self.data['time'] - intervention_start,
            0
        )
        
        # Prepare model
        X = self.data[['time', 'intervention', 'time_since_intervention']]
        X = sm.add_constant(X)
        y = self.data[self.health_col]
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Generate predictions and counterfactual
        predictions = model.predict(X)
        
        # Counterfactual (what would have happened without intervention)
        X_counter = X.copy()
        X_counter['intervention'] = 0
        X_counter['time_since_intervention'] = 0
        counterfactual = model.predict(X_counter)
        
        # Store results
        self.itsa_results = ITSAResults(
            baseline_level=model.params['const'],
            pre_trend=model.params['time'],
            level_change=model.params['intervention'],
            trend_change=model.params['time_since_intervention'],
            model=model,
            predictions=predictions,
            counterfactual=counterfactual,
            p_values={
                'baseline': model.pvalues['const'],
                'pre_trend': model.pvalues['time'],
                'level_change': model.pvalues['intervention'],
                'trend_change': model.pvalues['time_since_intervention']
            }
        )
        
        return self.itsa_results
    
    def seasonal_analysis(self, period: int = 12, model: str = 'additive') -> Dict:
        """
        Perform seasonal decomposition
        
        Parameters:
        -----------
        period : int
            Seasonal period (12 for monthly data)
        model : str
            'additive' or 'multiplicative'
        
        Returns:
        --------
        Dictionary containing seasonal analysis results
        """
        # Decompose PM2.5
        decomp_pm = seasonal_decompose(
            self.data[self.pm_col], 
            model=model, 
            period=period,
            extrapolate_trend='freq'
        )
        
        # Decompose health outcome
        decomp_health = seasonal_decompose(
            self.data[self.health_col],
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        
        # Calculate seasonal patterns
        pm_seasonal = decomp_pm.seasonal.groupby(decomp_pm.seasonal.index.month).mean()
        health_seasonal = decomp_health.seasonal.groupby(decomp_health.seasonal.index.month).mean()
        
        # Seasonal correlation
        seasonal_corr = np.corrcoef(pm_seasonal.values, health_seasonal.values)[0, 1]
        
        self.seasonal_decomposition = {
            'pm_decomposition': decomp_pm,
            'health_decomposition': decomp_health,
            'pm_seasonal_pattern': pm_seasonal,
            'health_seasonal_pattern': health_seasonal,
            'seasonal_correlation': seasonal_corr,
            'pm_trend': decomp_pm.trend,
            'health_trend': decomp_health.trend
        }
        
        return self.seasonal_decomposition
    
    def create_features(self, max_lag: int = 4, 
                       ma_windows: List[int] = [4, 8, 12]) -> pd.DataFrame:
        """
        Create time series features for machine learning
        
        Parameters:
        -----------
        max_lag : int
            Maximum lag periods to create
        ma_windows : list
            Moving average window sizes
        
        Returns:
        --------
        DataFrame with engineered features
        """
        df = self.data.copy()
        
        # Lag features
        for col in [self.pm_col, self.health_col, 'avgtemp', 'avghumidity']:
            if col in df.columns:
                for lag in range(1, max_lag + 1):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Moving averages
        for col in [self.pm_col, self.health_col]:
            if col in df.columns:
                for window in ma_windows:
                    df[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_ewm_{window}'] = df[col].ewm(span=window).mean()
        
        # Calendar features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Change features
        for col in [self.pm_col, self.health_col]:
            if col in df.columns:
                df[f'{col}_diff'] = df[col].diff()
                df[f'{col}_pct_change'] = df[col].pct_change()
                df[f'{col}_accel'] = df[col].diff().diff()
                df[f'{col}_increasing'] = (df[f'{col}_diff'] > 0).astype(int)
        
        self.data = df
        return df
    
    def build_ml_model(self, features: List[str] = None, 
                      test_size: float = 0.2,
                      model_type: str = 'random_forest') -> Dict:
        """
        Build and evaluate machine learning prediction model
        
        Parameters:
        -----------
        features : list
            Feature columns to use (if None, auto-selects)
        test_size : float
            Proportion of data for testing
        model_type : str
            Type of model ('random_forest', 'ridge', 'lasso')
        
        Returns:
        --------
        Dictionary with model performance metrics
        """
        # Prepare data
        data_clean = self.data.dropna()
        
        # Auto-select features if not provided
        if features is None:
            features = self._select_features(data_clean)
        
        X = data_clean[features]
        y = data_clean[self.health_col]
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select and train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        self.ml_model = {
            'model': model,
            'features': features,
            'scaler': scaler if model_type in ['ridge', 'lasso'] else None,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            },
            'predictions': y_pred,
            'test_actual': y_test
        }
        
        return self.ml_model
    
    def _select_features(self, data: pd.DataFrame, n_features: int = 10) -> List[str]:
        """
        Automatically select top features based on correlation and importance
        """
        # Exclude target and non-numeric columns
        feature_cols = [col for col in data.columns 
                       if col != self.health_col and data[col].dtype in ['float64', 'int64']]
        
        if not feature_cols:
            return []
        
        X = data[feature_cols]
        y = data[self.health_col]
        
        # Select based on correlation
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(n_features).index.tolist()
        
        return top_features
    
    def run_sarima(self, order: Tuple[int, int, int] = (1, 1, 1),
                   seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                   forecast_periods: int = 12) -> Dict:
        """
        Run SARIMA model for forecasting
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) parameters
        seasonal_order : tuple
            (P, D, Q, s) seasonal parameters
        forecast_periods : int
            Number of periods to forecast
        
        Returns:
        --------
        Dictionary with SARIMA results
        """
        # Split data for validation
        split_date = self.data.index[-forecast_periods]
        train_data = self.data[:split_date][self.health_col]
        test_data = self.data[split_date:][self.health_col]
        
        # Fit SARIMA model
        model = ARIMA(train_data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast = fitted_model.forecast(steps=len(test_data))
        forecast_full = fitted_model.forecast(steps=forecast_periods)
        forecast_ci = fitted_model.get_forecast(steps=forecast_periods).conf_int()
        
        # Calculate accuracy
        mae = mean_absolute_error(test_data, forecast[:len(test_data)])
        rmse = np.sqrt(mean_squared_error(test_data, forecast[:len(test_data)]))
        mape = np.mean(np.abs((test_data - forecast[:len(test_data)]) / test_data)) * 100
        
        return {
            'model': fitted_model,
            'forecast': forecast_full,
            'confidence_intervals': forecast_ci,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            },
            'residuals': fitted_model.resid
        }


class InterventionSimulator:
    """
    Simulate health impacts of interventions
    """
    
    def __init__(self, baseline_pm25: float, baseline_tb: float,
                 beta: float = 0.08, population: int = 1500000):
        """
        Initialize simulator
        
        Parameters:
        -----------
        baseline_pm25 : float
            Baseline PM2.5 level
        baseline_tb : float
            Baseline TB cases
        beta : float
            Concentration-response coefficient
        population : int
            Population size
        """
        self.baseline_pm25 = baseline_pm25
        self.baseline_tb = baseline_tb
        self.beta = beta
        self.population = population
        
    def simulate_scenario(self, pm25_reduction_pct: float,
                         implementation_years: int = 3,
                         cost_factor: float = 1.0,
                         time_horizon: int = 60) -> SimulationResults:
        """
        Simulate a single intervention scenario
        
        Parameters:
        -----------
        pm25_reduction_pct : float
            Percentage reduction in PM2.5
        implementation_years : int
            Years to implement
        cost_factor : float
            Cost multiplier relative to baseline
        time_horizon : int
            Months to simulate
        
        Returns:
        --------
        SimulationResults object
        """
        # Calculate health impact
        new_pm25 = self.baseline_pm25 * (1 - pm25_reduction_pct/100)
        pm25_change = new_pm25 - self.baseline_pm25
        
        # Concentration-response function
        relative_risk = np.exp(self.beta * (pm25_change / 10))
        new_tb_cases = self.baseline_tb * relative_risk
        cases_prevented_monthly = self.baseline_tb - new_tb_cases
        cases_prevented_total = cases_prevented_monthly * time_horizon
        
        # Economic analysis
        base_cost = 1000000  # $1M baseline
        total_cost = base_cost * cost_factor * implementation_years
        healthcare_savings = cases_prevented_total * 2000  # $2000 per case
        cost_per_case = total_cost / cases_prevented_total if cases_prevented_total > 0 else float('inf')
        
        # NPV calculation (simplified)
        discount_rate = 0.05
        annual_benefits = (cases_prevented_total / 5) * 2000  # Annual healthcare savings
        pv_benefits = sum([annual_benefits / (1 + discount_rate)**year 
                          for year in range(1, 11)])  # 10-year horizon
        npv = pv_benefits - total_cost
        bcr = pv_benefits / total_cost if total_cost > 0 else 0
        
        return SimulationResults(
            scenario_name="Custom Scenario",
            pm25_reduction=pm25_reduction_pct,
            new_pm25=new_pm25,
            cases_prevented_monthly=cases_prevented_monthly,
            cases_prevented_total=cases_prevented_total,
            relative_risk=relative_risk,
            cost_per_case=cost_per_case,
            benefit_cost_ratio=bcr,
            net_present_value=npv
        )
    
    def monte_carlo_simulation(self, scenarios: Dict[str, Dict],
                              n_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation for uncertainty analysis
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary of scenarios with parameters
        n_simulations : int
            Number of Monte Carlo iterations
        
        Returns:
        --------
        Dictionary with uncertainty results
        """
        results = {name: {'cases_prevented': [], 'cost_per_case': [], 'bcr': []}
                  for name in scenarios.keys()}
        
        for _ in range(n_simulations):
            # Sample uncertain parameters
            beta_sample = np.random.normal(self.beta, 0.02)
            beta_sample = np.clip(beta_sample, 0.04, 0.15)
            
            for name, params in scenarios.items():
                # Add implementation uncertainty
                actual_reduction = np.random.normal(
                    params['pm25_reduction'],
                    params['pm25_reduction'] * 0.2
                )
                actual_reduction = np.clip(actual_reduction, 0, params['pm25_reduction'] * 1.5)
                
                # Cost uncertainty
                cost_multiplier = np.random.uniform(0.8, 1.5)
                
                # Temporarily update beta
                original_beta = self.beta
                self.beta = beta_sample
                
                # Simulate
                result = self.simulate_scenario(
                    actual_reduction,
                    params.get('implementation_years', 3),
                    params.get('cost_factor', 1.0) * cost_multiplier
                )
                
                # Store results
                results[name]['cases_prevented'].append(result.cases_prevented_total)
                results[name]['cost_per_case'].append(result.cost_per_case)
                results[name]['bcr'].append(result.benefit_cost_ratio)
                
                # Restore beta
                self.beta = original_beta
        
        # Calculate confidence intervals
        summary = {}
        for name in scenarios.keys():
            summary[name] = {
                'cases_prevented_ci': np.percentile(results[name]['cases_prevented'], [2.5, 50, 97.5]),
                'cost_per_case_ci': np.percentile(
                    [c for c in results[name]['cost_per_case'] if c < 1e6], [2.5, 50, 97.5]
                ),
                'bcr_ci': np.percentile(results[name]['bcr'], [2.5, 50, 97.5])
            }
        
        return summary


class VisualizationHelper:
    """
    Helper class for creating visualizations
    """
    
    @staticmethod
    def plot_itsa_results(data: pd.DataFrame, itsa_results: ITSAResults,
                          intervention_date: str, save_path: str = None):
        """
        Plot ITSA analysis results
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot observed data
        ax.plot(data.index, data.iloc[:, 0], 'ko-', label='Observed', 
                markersize=4, alpha=0.6)
        
        # Plot fitted values
        ax.plot(data.index, itsa_results.predictions, 'b-', 
                label='ITSA Model', linewidth=2)
        
        # Plot counterfactual
        post_intervention = data.index >= pd.to_datetime(intervention_date)
        if any(post_intervention):
            ax.plot(data.index[post_intervention], 
                   itsa_results.counterfactual[post_intervention],
                   'r--', label='Counterfactual', linewidth=2)
        
        # Mark intervention
        ax.axvline(pd.to_datetime(intervention_date), color='red', 
                  linestyle=':', alpha=0.7, label='Intervention')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Outcome')
        ax.set_title('Interrupted Time Series Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_seasonal_decomposition(decomposition: Any, title: str = "Seasonal Decomposition"):
        """
        Plot seasonal decomposition results
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Original')
        axes[0].set_ylabel('Original')
        
        decomposition.trend.plot(ax=axes[1], title='Trend')
        axes[1].set_ylabel('Trend')
        
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        axes[2].set_ylabel('Seasonal')
        
        decomposition.resid.plot(ax=axes[3], title='Residual')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_intervention_comparison(simulation_results: Dict[str, SimulationResults]):
        """
        Plot comparison of intervention scenarios
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        scenarios = list(simulation_results.keys())
        
        # Extract data
        pm25_reductions = [r.pm25_reduction for r in simulation_results.values()]
        cases_prevented = [r.cases_prevented_total for r in simulation_results.values()]
        costs_per_case = [r.cost_per_case for r in simulation_results.values()]
        bcr_values = [r.benefit_cost_ratio for r in simulation_results.values()]
        
        # 1. Effectiveness plot
        ax1.scatter(pm25_reductions, cases_prevented, s=100, alpha=0.7)
        for i, name in enumerate(scenarios):
            ax1.annotate(name, (pm25_reductions[i], cases_prevented[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('PM2.5 Reduction (%)')
        ax1.set_ylabel('TB Cases Prevented')
        ax1.set_title('Intervention Effectiveness')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cost-effectiveness
        bars = ax2.bar(range(len(scenarios)), costs_per_case, alpha=0.7)
        ax2.set_ylabel('Cost per Case Prevented ($)')
        ax2.set_title('Cost-Effectiveness Analysis')
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Benefit-cost ratio
        bars = ax3.bar(range(len(scenarios)), bcr_values, alpha=0.7)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Benefit-Cost Ratio')
        ax3.set_title('Economic Viability')
        ax3.set_xticks(range(len(scenarios)))
        ax3.set_xticklabels(scenarios, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. NPV
        npv_values = [r.net_present_value/1e6 for r in simulation_results.values()]
        colors = ['green' if npv > 0 else 'red' for npv in npv_values]
        ax4.barh(range(len(scenarios)), npv_values, color=colors, alpha=0.7)
        ax4.set_xlabel('Net Present Value ($M)')
        ax4.set_title('10-Year NPV by Scenario')
        ax4.set_yticks(range(len(scenarios)))
        ax4.set_yticklabels(scenarios)
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Utility functions
def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> Tuple[bool, float]:
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test
    
    Returns:
    --------
    Tuple of (is_stationary, p_value)
    """
    result = adfuller(series.dropna())
    return result[1] < significance_level, result[1]


def calculate_forecast_accuracy(actual: pd.Series, forecast: pd.Series) -> Dict[str, float]:
    """
    Calculate various forecast accuracy metrics
    """
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def find_optimal_sarima_params(data: pd.Series, 
                               p_range: range = range(0, 3),
                               d_range: range = range(0, 2), 
                               q_range: range = range(0, 3),
                               P_range: range = range(0, 2),
                               D_range: range = range(0, 2),
                               Q_range: range = range(0, 2),
                               s: int = 12) -> Tuple:
    """
    Grid search for optimal SARIMA parameters
    
    Returns:
    --------
    Tuple of (best_order, best_seasonal_order, best_aic)
    """
    import itertools
    
    best_aic = float('inf')
    best_params = None
    
    for params in itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range):
        try:
            model = ARIMA(data, 
                         order=(params[0], params[1], params[2]),
                         seasonal_order=(params[3], params[4], params[5], s))
            fitted = model.fit(disp=0)
            
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_params = params
        except:
            continue
    
    if best_params:
        return (best_params[:3], best_params[3:] + (s,), best_aic)
    return None


def create_summary_report(analyzer: TimeSeriesHealthAnalyzer,
                         simulator: InterventionSimulator,
                         scenarios: Dict) -> pd.DataFrame:
    """
    Create a summary report of all analyses
    """
    report_data = []
    
    # ITSA results
    if analyzer.itsa_results:
        report_data.append({
            'Analysis': 'ITSA',
            'Metric': 'Level Change',
            'Value': f"{analyzer.itsa_results.level_change:.2f}",
            'Significance': f"p={analyzer.itsa_results.p_values['level_change']:.4f}"
        })
        report_data.append({
            'Analysis': 'ITSA',
            'Metric': 'Trend Change',
            'Value': f"{analyzer.itsa_results.trend_change:.3f}",
            'Significance': f"p={analyzer.itsa_results.p_values['trend_change']:.4f}"
        })
    
    # ML model results
    if analyzer.ml_model:
        report_data.append({
            'Analysis': 'ML Model',
            'Metric': 'R-squared',
            'Value': f"{analyzer.ml_model['metrics']['r2']:.3f}",
            'Significance': 'N/A'
        })
        report_data.append({
            'Analysis': 'ML Model',
            'Metric': 'MAE',
            'Value': f"{analyzer.ml_model['metrics']['mae']:.2f}",
            'Significance': 'N/A'
        })
    
    # Simulation results
    for name, result in scenarios.items():
        report_data.append({
            'Analysis': f'Scenario: {name}',
            'Metric': 'Cases Prevented',
            'Value': f"{result.cases_prevented_total:.0f}",
            'Significance': f"BCR={result.benefit_cost_ratio:.2f}"
        })
    
    return pd.DataFrame(report_data)