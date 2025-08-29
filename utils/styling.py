"""
Custom CSS Styling for Streamlit Application
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 0rem 0rem;
        }
        
        /* Custom fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Headers styling */
        h1 {
            color: #1e3a8a;
            font-weight: 700;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        
        h2 {
            color: #1e40af;
            font-weight: 600;
            margin-top: 30px;
        }
        
        h3 {
            color: #2563eb;
            font-weight: 600;
        }
        
        /* Metric cards */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            border: none;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            color: white;
            font-weight: 700;
            font-size: 2rem;
        }
        
        [data-testid="metric-container"] [data-testid="metric-label"] {
            color: rgba(255,255,255,0.9);
            font-weight: 600;
            font-size: 1rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1e3a8a 0%, #3730a3 100%);
        }
        
        .css-1d391kg .css-1l269bu {
            color: white;
        }
        
        /* Info boxes */
        .stAlert {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 15px;
        }
        
        /* Success boxes */
        .success-box {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Warning boxes */
        .warning-box {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Cards */
        .card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e5e7eb;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #f3f4f6;
            padding: 10px;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: 2px solid transparent;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
        }
        
        /* DataFrame styling */
        .dataframe {
            border: none !important;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .dataframe thead th {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px !important;
        }
        
        .dataframe tbody tr:hover {
            background: #f3f4f6 !important;
        }
    </style>
    """, unsafe_allow_html=True)