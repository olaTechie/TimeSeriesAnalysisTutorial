"""
Helper Functions for Streamlit Application
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

def create_gauge_chart(value, title, max_value=100):
    """Create a beautiful gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': max_value * 0.5},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value*0.25], 'color': '#10b981'},
                {'range': [max_value*0.25, max_value*0.5], 'color': '#3b82f6'},
                {'range': [max_value*0.5, max_value*0.75], 'color': '#f59e0b'},
                {'range': [max_value*0.75, max_value], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_time_series_plot(data, x_col, y_cols, title="Time Series"):
    """Create time series plot with multiple lines"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[col],
            mode='lines+markers',
            name=col.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title="Value",
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    return fig

def calculate_health_impact(pm25_baseline, pm25_reduced, cr_pneumonia=1.12, cr_tb=1.08):
    """Calculate health impact based on PM2.5 reduction"""
    pm25_change = pm25_reduced - pm25_baseline
    
    # Calculate relative risks
    pneumonia_rr = cr_pneumonia ** (pm25_change / 10)
    tb_rr = cr_tb ** (pm25_change / 10)
    
    # Calculate percentage changes
    pneumonia_change = (pneumonia_rr - 1) * 100
    tb_change = (tb_rr - 1) * 100
    
    return {
        'pneumonia_rr': pneumonia_rr,
        'tb_rr': tb_rr,
        'pneumonia_change_pct': pneumonia_change,
        'tb_change_pct': tb_change
    }

def format_number(value, decimals=0):
    """Format large numbers with commas"""
    if pd.isna(value):
        return "N/A"
    if decimals == 0:
        return f"{value:,.0f}"
    else:
        return f"{value:,.{decimals}f}"

def calculate_roi(cases_prevented, cost_factor, baseline_cost=1000000):
    """Calculate return on investment"""
    benefit = cases_prevented * 1000  # Assumed cost per case
    cost = baseline_cost * cost_factor
    roi = ((benefit - cost) / cost) * 100 if cost > 0 else 0
    return roi

def generate_download_link(df, filename="data.csv"):
    """Generate CSV download link for dataframe"""
    csv = df.to_csv(index=False)
    return csv

def get_scenario_params():
    """Get predefined scenario parameters"""
    return {
        "Baseline (No Intervention)": {
            "reduction": 0, 
            "cost": 1.0, 
            "time": 0,
            "sustainability": 0.5
        },
        "Traffic Carbon Emission Controls": {
            "reduction": 25, 
            "cost": 2.5, 
            "time": 12,
            "sustainability": 0.7
        },
        "Urban Green Space Expansion": {
            "reduction": 15, 
            "cost": 3.0, 
            "time": 24,
            "sustainability": 0.9
        },
        "Road Paving (Dust Reduction)": {
            "reduction": 20, 
            "cost": 4.0, 
            "time": 18,
            "sustainability": 0.8
        },
        "Combined: Traffic + Green Spaces": {
            "reduction": 35, 
            "cost": 5.0, 
            "time": 24,
            "sustainability": 0.85
        },
        "All Interventions Combined": {
            "reduction": 45, 
            "cost": 8.0, 
            "time": 36,
            "sustainability": 0.9
        }
    }

def create_correlation_heatmap(data, variables=None, title="Correlation Matrix"):
    """Create correlation heatmap"""
    if variables is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        variables = [col for col in numeric_cols if not col.endswith('_lag')][:10]
    
    corr_matrix = data[variables].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title=title,
        aspect='auto'
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white'
    )
    
    return fig

def sigmoid_implementation(total_months, implementation_time):
    """Generate sigmoid curve for gradual implementation"""
    if implementation_time == 0:
        return np.ones(total_months)
    
    x = np.linspace(0, 10, total_months)
    center = 5 * (implementation_time / 12)
    sigmoid = 1 / (1 + np.exp(-(x - center)))
    return sigmoid