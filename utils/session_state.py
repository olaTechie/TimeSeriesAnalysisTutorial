"""
Session State Management for Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Data state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Model state
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    
    if 'model_features' not in st.session_state:
        st.session_state.model_features = []
    
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    
    # Simulation state
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = 'baseline'
    
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = {}
    
    # Analysis state
    if 'sensitivity_params' not in st.session_state:
        st.session_state.sensitivity_params = {
            'cr_pneumonia': 12.0,
            'cr_tb': 8.0,
            'baseline_disease_rate': 1.0,
            'cost_per_case': 1000,
            'discount_rate': 3.0,
            'implementation_efficiency': 80
        }
    
    if 'spatial_results' not in st.session_state:
        st.session_state.spatial_results = None
    
    # Report state
    if 'report_config' not in st.session_state:
        st.session_state.report_config = {
            'type': 'Executive Summary',
            'format': 'PDF',
            'include_visuals': True,
            'sections': ['Executive Summary', 'Recommendations']
        }

def get_data():
    """Get or load data"""
    if not st.session_state.data_loaded:
        st.session_state.data = load_sample_data()
        st.session_state.data_loaded = True
    return st.session_state.data

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2020-01', '2024-12', freq='M')
    counties = ['Kampala', 'Nakawa', 'Makindye', 'Kawempe', 'Rubaga']
    
    data = []
    for date in dates:
        for county in counties:
            # Create correlated data
            base_pm25 = 35 + np.random.randint(-5, 5)
            pm25 = np.random.normal(base_pm25, 5)
            
            # Health outcomes correlated with PM2.5
            pneumonia_base = int(pm25 * 1.5) + np.random.randint(-5, 5)
            tb_base = int(pm25 * 0.6) + np.random.randint(-3, 3)
            
            data.append({
                'year_month': date,
                'county': county,
                'avg_pm2_5_calibrated': max(0, pm25),
                'total_pneumonia': max(0, np.random.poisson(pneumonia_base)),
                'total_tb': max(0, np.random.poisson(tb_base)),
                'temperature': np.random.normal(25, 3),
                'humidity': np.random.normal(70, 10),
                'population_density': np.random.randint(1000, 5000),
                'month': date.month,
                'year': date.year,
                'quarter': date.quarter
            })
    
    df = pd.DataFrame(data)
    
    # Add lag features
    for lag in [1, 2, 3]:
        df[f'pm25_lag_{lag}'] = df.groupby('county')['avg_pm2_5_calibrated'].shift(lag)
        df[f'pneumonia_lag_{lag}'] = df.groupby('county')['total_pneumonia'].shift(lag)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(0)
    
    return df

def save_to_session(key, value):
    """Save value to session state"""
    st.session_state[key] = value

def get_from_session(key, default=None):
    """Get value from session state"""
    return st.session_state.get(key, default)