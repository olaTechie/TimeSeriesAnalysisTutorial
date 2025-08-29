"""
Time Series Modeling Module for Health Impact Prediction
This module implements time series classification models for predicting health outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesModeler:
    """Time series classification for health impact prediction"""
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'total_pneumonia'):
        """
        Initialize time series modeler
        
        Args:
            data: DataFrame with time series data
            target_column: Column to use as prediction target
        """
        self.data = data.copy()
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = {}
        logger.info(f"TimeSeriesModeler initialized with {len(data)} records, target: {target_column}")
    
    def prepare_time_series_data(self, window_size: int = 12,
                                 features: Optional[List[str]] = None,
                                 target_type: str = 'binary') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for time series classification
        
        Args:
            window_size: Size of the time window for features
            features: List of features to use (None for default)
            target_type: 'binary' for high/low, 'multiclass' for categories
            
        Returns:
            Tuple of (X, y) arrays for modeling
        """
        # Sort by county and time
        sorted_data = self.data.sort_values(['county', 'year_month'])
        
        # Default features if not specified
        if features is None:
            features = ['avg_pm2_5_calibrated']
            if 'month' in sorted_data.columns:
                features.append('month')
            if 'quarter' in sorted_data.columns:
                features.append('quarter')
            
            # Add lag features if available
            lag_features = [col for col in sorted_data.columns if 'lag' in col]
            features.extend(lag_features[:3])  # Use up to 3 lag features
        
        X_list = []
        y_list = []
        
        for county in sorted_data['county'].unique():
            county_data = sorted_data[sorted_data['county'] == county].reset_index(drop=True)
            
            if len(county_data) < window_size + 1:
                continue
            
            # Create sliding windows
            for i in range(len(county_data) - window_size):
                window = county_data.iloc[i:i+window_size]
                
                # Extract features for the window
                window_features = []
                for feature in features:
                    if feature in window.columns:
                        window_features.extend(window[feature].values)
                
                X_list.append(window_features)
                
                # Create target
                target_value = county_data[self.target_column].iloc[i + window_size]
                
                if target_type == 'binary':
                    # Binary classification: above/below median
                    threshold = county_data[self.target_column].median()
                    y_label = 1 if target_value > threshold else 0
                elif target_type == 'multiclass':
                    # Multiclass: quartiles
                    quartiles = county_data[self.target_column].quantile([0.25, 0.5, 0.75])
                    if target_value <= quartiles[0.25]:
                        y_label = 0  # Low
                    elif target_value <= quartiles[0.5]:
                        y_label = 1  # Medium-low
                    elif target_value <= quartiles[0.75]:
                        y_label = 2  # Medium-high
                    else:
                        y_label = 3  # High
                else:
                    y_label = target_value  # Regression
                
                y_list.append(y_label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1] if len(X) > 0 else 0} features")
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray,
                    models_to_train: Optional[List[str]] = None,
                    cv_splits: int = 3) -> Dict:
        """
        Train multiple time series models
        
        Args:
            X: Feature array
            y: Target array
            models_to_train: List of model names to train
            cv_splits: Number of cross-validation splits
            
        Returns:
            Dictionary with model results
        """
        if len(X) == 0:
            logger.warning("No data available for training")
            return {}
        
        # Default models
        if models_to_train is None:
            models_to_train = ['random_forest', 'gradient_boosting', 'svm']
        
        # Initialize models
        model_dict = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Try to use sktime models if available
        try:
            from sktime.classification.interval_based import TimeSeriesForestClassifier
            from sktime.classification.kernel_based import RocketClassifier
            
            # Reshape for sktime
            X_sktime = X.reshape(X.shape[0], 1, X.shape[1])
            
            model_dict.update({
                'ts_forest': TimeSeriesForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),
                'rocket': RocketClassifier(
                    num_kernels=1000,
                    random_state=42
                )
            })
            use_sktime = True
        except ImportError:
            logger.info("sktime not available, using sklearn models only")
            use_sktime = False
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        results = {}
        
        for model_name in models_to_train:
            if model_name not in model_dict:
                logger.warning(f"Model {model_name} not available")
                continue
            
            model = model_dict[model_name]
            logger.info(f"Training {model_name}...")
            
            scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
                # Use appropriate data format
                if use_sktime and model_name in ['ts_forest', 'rocket']:
                    X_train = X_sktime[train_idx]
                    X_test = X_sktime[test_idx]
                else:
                    X_train = X_scaled[train_idx]
                    X_test = X_scaled[test_idx]
                
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                fold_scores = {
                    'fold': fold + 1,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred, average='weighted'),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted')
                }
                
                # Add ROC-AUC for binary classification
                if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fold_scores['roc_auc'] = roc_auc_score(y_test, y_proba)
                
                scores.append(fold_scores)
            
            # Train final model on all data
            if use_sktime and model_name in ['ts_forest', 'rocket']:
                model.fit(X_sktime, y)
            else:
                model.fit(X_scaled, y)
            
            self.models[model_name] = model
            
            # Calculate average scores
            avg_scores = {
                'model': model_name,
                'avg_accuracy': np.mean([s['accuracy'] for s in scores]),
                'avg_f1': np.mean([s['f1'] for s in scores]),
                'avg_precision': np.mean([s['precision'] for s in scores]),
                'avg_recall': np.mean([s['recall'] for s in scores]),
                'std_accuracy': np.std([s['accuracy'] for s in scores]),
                'std_f1': np.std([s['f1'] for s in scores]),
                'fold_scores': scores
            }
            
            if 'roc_auc' in scores[0]:
                avg_scores['avg_roc_auc'] = np.mean([s['roc_auc'] for s in scores])
            
            results[model_name] = avg_scores
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
        
        # Identify best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['avg_f1'])
            self.best_model = self.models[best_model_name]
            logger.info(f"Best model: {best_model_name} (F1: {results[best_model_name]['avg_f1']:.3f})")
        
        self.results = results
        return results
    
    def predict(self, X_new: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions with trained model
        
        Args:
            X_new: New data for prediction
            model_name: Name of model to use (None for best model)
            
        Returns:
            Array of predictions
        """
        if model_name is None:
            model = self.best_model
            if model is None:
                raise ValueError("No trained models available")
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        # Scale features
        scaler = StandardScaler()
        scaler.fit(X_new)
        X_scaled = scaler.transform(X_new)
        
        return model.predict(X_scaled)
    
    def forecast_health_burden(self, pm25_forecast: pd.DataFrame,
                              window_size: int = 12) -> pd.DataFrame:
        """
        Forecast health burden based on PM2.5 projections
        
        Args:
            pm25_forecast: DataFrame with PM2.5 forecasts
            window_size: Window size used in training
            
        Returns:
            DataFrame with health burden forecasts
        """
        if self.best_model is None:
            raise ValueError("Train models first before forecasting")
        
        forecasts = []
        
        for county in pm25_forecast['county'].unique():
            county_data = pm25_forecast[pm25_forecast['county'] == county]
            
            if len(county_data) < window_size:
                continue
            
            # Prepare features
            features = ['avg_pm2_5_calibrated']
            if 'month' in county_data.columns:
                features.append('month')
            
            X_forecast = []
            for i in range(len(county_data) - window_size + 1):
                window = county_data.iloc[i:i+window_size]
                window_features = []
                for feature in features:
                    if feature in window.columns:
                        window_features.extend(window[feature].values)
                X_forecast.append(window_features)
            
            if X_forecast:
                X_forecast = np.array(X_forecast)
                predictions = self.predict(X_forecast)
                
                for i, pred in enumerate(predictions):
                    forecasts.append({
                        'county': county,
                        'date': county_data.iloc[i + window_size - 1]['year_month'],
                        'predicted_burden': pred,
                        'pm25_level': county_data.iloc[i + window_size - 1]['avg_pm2_5_calibrated']
                    })
        
        return pd.DataFrame(forecasts)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison table of all trained models
        
        Returns:
            DataFrame with model comparison metrics
        """
        if not self.results:
            logger.warning("No results available. Train models first.")
            return pd.DataFrame()
        
        comparison = []
        for model_name, results in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': f"{results['avg_accuracy']:.3f} ± {results['std_accuracy']:.3f}",
                'F1 Score': f"{results['avg_f1']:.3f} ± {results['std_f1']:.3f}",
                'Precision': f"{results['avg_precision']:.3f}",
                'Recall': f"{results['avg_recall']:.3f}",
                'ROC-AUC': f"{results.get('avg_roc_auc', 'N/A'):.3f}" if 'avg_roc_auc' in results else 'N/A'
            })
        
        return pd.DataFrame(comparison)
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance from trained models
        
        Args:
            model_name: Model to get importance from (None for best model)
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            # Find best model with feature importance
            for name in self.feature_importance.keys():
                if self.models[name] == self.best_model:
                    model_name = name
                    break
        
        if model_name not in self.feature_importance:
            logger.warning(f"No feature importance available for {model_name}")
            return pd.DataFrame()
        
        importance = self.feature_importance[model_name]
        
        # Create feature names (assuming standard structure)
        feature_names = []
        window_size = len(importance) // len(['avg_pm2_5_calibrated', 'month', 'quarter'])
        
        for i in range(len(importance)):
            feature_idx = i // window_size
            time_idx = i % window_size
            
            if feature_idx == 0:
                feature_names.append(f'PM2.5_t-{window_size-time_idx-1}')
            elif feature_idx == 1:
                feature_names.append(f'Month_t-{window_size-time_idx-1}')
            else:
                feature_names.append(f'Quarter_t-{window_size-time_idx-1}')
        
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importance)],
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_models(self, output_dir: str = './models'):
        """
        Save trained models to disk
        
        Args:
            output_dir: Directory to save models
        """
        import pickle
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = output_path / f'{model_name}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save results
        results_file = output_path / 'model_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Saved model results to {results_file}")
    
    def load_models(self, model_dir: str = './models'):
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
        """
        import pickle
        from pathlib import Path
        
        model_path = Path(model_dir)
        
        # Load models
        for model_file in model_path.glob('*_model.pkl'):
            model_name = model_file.stem.replace('_model', '')
            with open(model_file, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            logger.info(f"Loaded {model_name} from {model_file}")
        
        # Load results
        results_file = model_path / 'model_results.pkl'
        if results_file.exists():
            with open(results_file, 'rb') as f:
                self.results = pickle.load(f)
            logger.info(f"Loaded model results from {results_file}")