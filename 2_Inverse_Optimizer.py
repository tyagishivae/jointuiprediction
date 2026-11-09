import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna
from pathlib import Path
from datetime import datetime

# Custom CSS (same theme as main page)
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #ffffff;
    }
    
    /* Complete right panel area */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    
    /* Main content area - right panel */
    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
    }
    
    /* Title styling */
    .title-main {
        text-align: center;
        color: #003366;
        font-size: 2.5em;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 1.1em;
        margin-bottom: 30px;
        font-style: italic;
    }
            
    .sidebar-section-title{
        color: #ffffff !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.8px;
        margin-top: 10px;
        margin-bottom: 10px;
        opacity: 1 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #003366;
        margin-bottom: 10px;
    }
    
    .metric-label {
        color: #666666;
        font-size: 0.9em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #003366;
        font-size: 2em;
        font-weight: 700;
        margin-top: 8px;
    }
    
    .verdict-box {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 20px 0;
    }
    
    .verdict-excellent {
        border: 3px solid #28a745;
        background-color: #f0f9f6;
    }
    
    .verdict-good {
        border: 3px solid #ffc107;
        background-color: #fffbf0;
    }
    
    .verdict-moderate {
        border: 3px solid #fd7e14;
        background-color: #fff9f5;
    }
    
    .verdict-text {
        font-size: 2em;
        font-weight: 700;
        margin-top: 10px;
    }
    
    /* Section headers */
    .section-header {
        color: #003366;
        font-size: 1.4em;
        font-weight: 700;
        border-bottom: 3px solid #003366;
        padding-bottom: 10px;
        margin-top: 25px;
        margin-bottom: 20px;
    }
    
    /* Text visibility */
    .stMarkdown {
        color: #333333 !important;
    }

    /* Streamlit metric text fix */
    div[data-testid="stMetric"] label {
        color: #003366 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #111111 !important;
        font-weight: 800 !important;
    }

    /* Metric delta if shown */
    div[data-testid="stMetricDelta"] {
        color: #28a745 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Constants
MODEL_DIR = Path("models")

# Load models with caching
@st.cache_resource
def load_models():
    """Load all trained models and feature order."""
    try:
        models = {
            'defect_classifier': joblib.load(MODEL_DIR / 'defect_classifier.pkl'),
            'YieldStrength_MPa': joblib.load(MODEL_DIR / 'YieldStrength_MPa.pkl'),
            'UTS_MPa': joblib.load(MODEL_DIR / 'UTS_MPa.pkl'),
            'Hardness_HRA': joblib.load(MODEL_DIR / 'Hardness_HRA.pkl'),
            'MeltPoolDepth_um': joblib.load(MODEL_DIR / 'MeltPoolDepth_um.pkl'),
            'MeltPoolWidth_um': joblib.load(MODEL_DIR / 'MeltPoolWidth_um.pkl'),
            'RelativeDensity_frac': joblib.load(MODEL_DIR / 'RelativeDensity_frac.pkl'),
            # 'Elongation_pct': joblib.load(MODEL_DIR / 'Elongation_pct.pkl'),
        }
        
        feature_order = joblib.load(MODEL_DIR / 'feature_order.pkl')
        return models, feature_order
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def make_input(P_W, v_mm_per_s, h_mm, t_mm, feature_order):
    """Create input dataframe in exact feature order."""
    values = {
        "P_W": P_W,
        "v_mm_per_s": v_mm_per_s,
        "h_mm": h_mm,
        "t_mm": t_mm
    }
    
    row = {col: values[col] for col in feature_order}
    return pd.DataFrame([row])

def objective(trial, models, feature_order, config):
    """Optuna objective function for parameter optimization."""
    
    # Suggest parameters
    P_W = trial.suggest_float("P_W", config['P_min'], config['P_max'])
    v_mm_per_s = trial.suggest_float("v_mm_per_s", config['v_min'], config['v_max'])
    h_mm = trial.suggest_float("h_mm", config['h_min'], config['h_max'])
    t_mm = trial.suggest_float("t_mm", config['t_min'], config['t_max'])
    
    # Make prediction
    X = make_input(P_W, v_mm_per_s, h_mm, t_mm, feature_order)
    
    try:
        # Defect prediction with probability
        try:
            probs = models['defect_classifier'].predict_proba(X)[0]
            defect_risk = 1 - np.max(probs)
        except:
            defect_pred = int(models['defect_classifier'].predict(X)[0])
            defect_risk = 0 if defect_pred == 1 else 0.5
        
        # Property predictions
        ys = float(models['YieldStrength_MPa'].predict(X)[0])
        uts = float(models['UTS_MPa'].predict(X)[0])
        hard = float(models['Hardness_HRA'].predict(X)[0])
        depth = float(models['MeltPoolDepth_um'].predict(X)[0])
        width = float(models['MeltPoolWidth_um'].predict(X)[0])
        density = float(models['RelativeDensity_frac'].predict(X)[0]) * 100
        # elong = float(models['Elongation_pct'].predict(X)[0])
        
        # Scoring logic (weighted for target optimization)
        score = 0
        
        # Base properties
        score += uts * 1.0
        score += ys * 0.8
        # score += hard * 5
        # score += elong * 25
        score -= abs(hard - config['target_hardness']) * 20
        score -= abs(density - config['target_density']) * 40
        
        # Defect penalty
        score -= defect_risk * 1500
        
        # Target penalties
        if uts < config['target_uts']:
            score -= (config['target_uts'] - uts) * 8
        
        if ys < config['target_ys']:
            score -= (config['target_ys'] - ys) * 8
        
        # if elong < config['min_elong']:
        #     score -= (config['min_elong'] - elong) * 100
        
        # Meltpool constraints
        if depth < config['min_depth']:
            score -= (config['min_depth'] - depth) * 40
        
        if depth > config['max_depth']:
            score -= (depth - config['max_depth']) * 40
        
        if width < config['min_width']:
            score -= (config['min_width'] - width) * 25
        
        if width > config['max_width']:
            score -= (config['max_width'] - width) * 25
        
        return -score  # Minimize negative score
        
    except Exception as e:
        return float('inf')

def evaluate_parameters(P_W, v_mm_per_s, h_mm, t_mm, models, feature_order):
    """Evaluate parameters and return predictions."""
    X = make_input(P_W, v_mm_per_s, h_mm, t_mm, feature_order)
    
    try:
        probs = models['defect_classifier'].predict_proba(X)[0]
        defect_risk = 1 - np.max(probs)
        defect_class = int(np.argmax(probs))
    except:
        defect_pred = int(models['defect_classifier'].predict(X)[0])
        defect_class = defect_pred
        defect_risk = 0 if defect_pred == 1 else 0.5
    
    results = {
        'YieldStrength_MPa': float(models['YieldStrength_MPa'].predict(X)[0]),
        'UTS_MPa': float(models['UTS_MPa'].predict(X)[0]),
        'Hardness_HRA': float(models['Hardness_HRA'].predict(X)[0]),
        # 'Elongation_pct': float(models['Elongation_pct'].predict(X)[0]),
        'RelativeDensity_pct': float(models['RelativeDensity_frac'].predict(X)[0]) * 100,
        'MeltPoolWidth_um': float(models['MeltPoolWidth_um'].predict(X)[0]),
        'MeltPoolDepth_um': float(models['MeltPoolDepth_um'].predict(X)[0]),
        'DefectClass': defect_class,
        'DefectRisk': defect_risk,
    }
    
    return results

def calculate_verdict(results, config):
    """Determine quality verdict based on results."""
    score = 0
    
    # Check targets
    if results['UTS_MPa'] >= config['target_uts']:
        score += 30
    elif results['UTS_MPa'] >= config['target_uts'] - 50:
        score += 20
    else:
        score += 10
    
    if results['YieldStrength_MPa'] >= config['target_ys']:
        score += 20
    elif results['YieldStrength_MPa'] >= config['target_ys'] - 40:
        score += 10
    else:
        score += 5
    
    # if results['Elongation_pct'] >= config['min_elong']:
    #     score += 20
    # elif results['Elongation_pct'] >= config['min_elong'] - 2:
    #     score += 10
    # else:
    #     score += 5
    
    # Defect check
    if results['DefectClass'] == 1:  # Stable
        score += 30
    elif results['DefectClass'] == 0:  # LOF
        score -= 10
    else:  # Keyhole
        score -= 20
    
    # Relative density target check
    if abs(results['RelativeDensity_pct'] - config['target_density']) <= 0.3:
        score += 40
    elif abs(results['RelativeDensity_pct'] - config['target_density']) <= 1.0:
        score += 25
    else:
        score += 10
        
    # Meltpool check
    if (config['min_width'] <= results['MeltPoolWidth_um'] <= config['max_width'] and
        config['min_depth'] <= results['MeltPoolDepth_um'] <= config['max_depth']):
        score += 10
    
    if score >= 85:
        return "Excellent", "#28a745"
    elif score >= 70:
        return "Good", "#ffc107"
    else:
        return "Moderate", "#fd7e14"

def show_metric_card(label, value, unit=""):
    """Display a metric card."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.2f} {unit}</div>
        </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<div class="title-main">INVERSE OPTIMIZER</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Automated LPBF Parameter Design for Target Properties</div>', unsafe_allow_html=True)
    
    # Load models
    models, feature_order = load_models()
    
    if models is None or feature_order is None:
        st.error("Failed to load models. Please check the Models folder.")
        return
    
    # Sidebar: Target specifications
    st.sidebar.markdown(
        """
        <div class="sidebar-section-title">
            TARGET SPECIFICATIONS
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    
    target_uts = st.sidebar.slider(
        "Target UTS (MPa)",
        min_value=900,
        max_value=1200,
        value=1080,
        step=10,
        help="Desired Ultimate Tensile Strength"
    )
    
    target_ys = st.sidebar.slider(
        "Target Yield Strength (MPa)",
        min_value=700,
        max_value=950,
        value=800,
        step=10,
        help="Desired Yield Strength"
    )

    target_hardness = st.sidebar.slider(
        "Target Hardness (HRA)",
        min_value=30.0,
        max_value=70.0,
        value=45.0,
        step=0.5,
        help="Desired hardness"
    )

    target_density = st.sidebar.slider(
        "Target Relative Density (%)",
        min_value=95.0,
        max_value=100.0,
        value=99.5,
        step=0.01,
        help="Desired relative density"
    )
    
    # min_elong = st.sidebar.slider(
    #     "Minimum Elongation (%)",
    #     min_value=5.0,
    #     max_value=25.0,
    #     value=15.0,
    #     step=0.5,
    #     help="Minimum acceptable elongation"
    # )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-title">MELTPOOL CONSTRAINTS</div>', unsafe_allow_html=True)
    
    col_w1, col_w2 = st.sidebar.columns(2)
    with col_w1:
        min_width = st.number_input(
            "Min Width (µm)",
            min_value=50,
            max_value=200,
            value=80,
            step=5
        )
    with col_w2:
        max_width = st.number_input(
            "Max Width (µm)",
            min_value=100,
            max_value=300,
            value=200,
            step=5
        )
    
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        min_depth = st.number_input(
            "Min Depth (µm)",
            min_value=20,
            max_value=100,
            value=50,
            step=5
        )
    with col_d2:
        max_depth = st.number_input(
            "Max Depth (µm)",
            min_value=50,
            max_value=200,
            value=140,
            step=5
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-title">OPTIMIZATION SETTINGS</div>', unsafe_allow_html=True)
    
    n_trials = st.sidebar.slider(
        "Optimization Trials",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of parameter combinations to test"
    )
    
    st.sidebar.markdown("---")
    
    # Optimize button
    optimize_clicked = st.sidebar.button("RUN OPTIMIZATION", use_container_width=True)
    
    # Configuration
    config = {
        'target_uts': target_uts,
        'target_ys': target_ys,
        # 'min_elong': min_elong,
        'target_hardness': target_hardness,
        'target_density': target_density,
        'min_width': min_width,
        'max_width': max_width,
        'min_depth': min_depth,
        'max_depth': max_depth,
        'P_min': 150,
        'P_max': 370,
        'v_min': 350,
        'v_max': 1772,
        'h_min': 0.09,
        'h_max': 0.12,
        't_min': 0.03,
        't_max': 0.05,
    }
    
    # Run optimization
    if optimize_clicked:
        with st.spinner("🔬 Running intelligent parameter search..."):
            progress_bar = st.progress(0)
            
            # Create study
            study = optuna.create_study(direction="minimize")
            
            # Optimize with progress callback
            def callback(study, trial):
                progress_bar.progress(min(trial.number / n_trials, 1.0))
            
            study.optimize(
                lambda trial: objective(trial, models, feature_order, config),
                n_trials=n_trials,
                callbacks=[callback]
            )
        
        progress_bar.progress(1.0)
        
        # Get best parameters
        best_params = study.best_params
        
        st.session_state.inverse_best_params = best_params
        st.session_state.inverse_config = config
        st.session_state.inverse_show_results = True
        st.session_state.inverse_study = study
    
    # Display results
    if st.session_state.get('inverse_show_results', False) and st.session_state.inverse_best_params is not None:
        best_params = st.session_state.inverse_best_params
        
        # Evaluate best parameters
        results = evaluate_parameters(
            best_params['P_W'],
            best_params['v_mm_per_s'],
            best_params['h_mm'],
            best_params['t_mm'],
            models,
            feature_order
        )
        
        verdict, verdict_color = calculate_verdict(results, config)
        
        # SECTION 1: Recommended Parameters
        st.markdown('<div class="section-header">RECOMMENDED PARAMETERS</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_metric_card("Power", best_params['P_W'], "W")
        with col2:
            show_metric_card("Scan Speed", best_params['v_mm_per_s'], "mm/s")
        with col3:
            show_metric_card("Hatch Spacing", best_params['h_mm'], "mm")
        with col4:
            show_metric_card("Layer Thickness", best_params['t_mm'], "mm")
        
        # SECTION 2: Predicted Performance
        st.markdown('<div class="section-header">PREDICTED PERFORMANCE</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_metric_card("Yield Strength", results['YieldStrength_MPa'], "MPa")
        with col2:
            show_metric_card("UTS", results['UTS_MPa'], "MPa")
        with col3:
            show_metric_card("Hardness", results['Hardness_HRA'], "HRA")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_metric_card("Relative Density", results['RelativeDensity_pct'], "%")
        with col2:
            show_metric_card("Meltpool Width", results['MeltPoolWidth_um'], "µm")
        with col3:
            show_metric_card("Meltpool Depth", results['MeltPoolDepth_um'], "µm")
        
        # col1, col2, col3 = st.columns(3)

        # with col1:
        #     show_metric_card("Hardness", results['Hardness_HRA'], "HRA")

        # with col2:
        #     show_metric_card("Meltpool Width", results['MeltPoolWidth_um'], "µm")

        # with col3:
        #     show_metric_card("Meltpool Depth", results['MeltPoolDepth_um'], "µm")

        # SECTION 3: Quality Verdict
        st.markdown('<div class="section-header">QUALITY VERDICT</div>', unsafe_allow_html=True)
        
        verdict_class = "verdict-excellent" if verdict == "Excellent" else ("verdict-good" if verdict == "Good" else "verdict-moderate")
        
        st.markdown(f"""
            <div class="verdict-box {verdict_class}">
                <div style="font-size: 1.2em; color: #666;">Overall Assessment</div>
                <div class="verdict-text" style="color: {verdict_color};">{verdict}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Defect status
        defect_mapping = {0: "Lack of Fusion", 1: "Stable", 2: "Keyhole"}
        defect_status = defect_mapping.get(results['DefectClass'], "Unknown")
        
        st.metric("Predicted Defect Class", defect_status)
        
        # SECTION 4: Export Results
        st.markdown('<div class="section-header">EXPORT RESULTS</div>', unsafe_allow_html=True)
        
        # Create export dataframe
        export_data = {
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Power_W': [best_params['P_W']],
            'ScanSpeed_mm_per_s': [best_params['v_mm_per_s']],
            'HatchSpacing_mm': [best_params['h_mm']],
            'LayerThickness_mm': [best_params['t_mm']],
            'YieldStrength_MPa': [results['YieldStrength_MPa']],
            'UTS_MPa': [results['UTS_MPa']],
            'Hardness_HRA': [results['Hardness_HRA']],
            'RelativeDensity_pct': [results['RelativeDensity_pct']],
            # 'Elongation_pct': [results['Elongation_pct']],
            'MeltPoolWidth_um': [results['MeltPoolWidth_um']],
            'MeltPoolDepth_um': [results['MeltPoolDepth_um']],
            'DefectClass': [defect_status],
            'DefectRisk': [results['DefectRisk']],
            'QualityVerdict': [verdict],
        }
        
        export_df = pd.DataFrame(export_data)
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Optimization Report (CSV)",
            data=csv_data,
            file_name=f"HAYNES282_InverseOptimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Optimization details
        st.markdown('<div class="section-header">OPTIMIZATION DETAILS</div>', unsafe_allow_html=True)
        
        trials_df = st.session_state.inverse_study.trials_dataframe()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trials", len(trials_df))
        with col2:
            st.metric("Best Trial #", st.session_state.inverse_study.best_trial.number)
        with col3:
            st.metric("Best Score", f"{-st.session_state.inverse_study.best_value:.2f}")
        
        # Only show available columns
        available_cols = ['number', 'value', 'state']
        for col in ['P_W', 'v_mm_per_s', 'h_mm', 't_mm']:
            if col in trials_df.columns:
                available_cols.append(col)
        st.dataframe(trials_df[available_cols].tail(10), use_container_width=True)
    
    else:
        # Initial state message
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                ### Welcome to Inverse Optimizer
                
                This tool uses intelligent parameter search to find LPBF process conditions
                that achieve your desired material properties.
                
                **How it works:**
                
                1. **Define targets** - Set your desired properties in the sidebar
                2. **Configure constraints** - Specify acceptable meltpool geometry ranges
                3. **Run optimization** - Click the button to search 50-500+ parameter combinations
                4. **Review recommendations** - Get the best parameter set with predicted performance
                
                **What you get:**
                - Optimal Power, Scan Speed, Hatch Spacing, Layer Thickness
                - Predicted mechanical properties
                - Meltpool geometry
                - Quality verdict (Excellent / Good / Moderate)
                - Exportable CSV report
                
                ---
                
                **Start by adjusting the targets in the sidebar and clicking RUN OPTIMIZATION!**
            """)

if __name__ == "__main__":
    # Initialize session state
    if 'inverse_show_results' not in st.session_state:
        st.session_state.inverse_show_results = False
    if 'inverse_best_params' not in st.session_state:
        st.session_state.inverse_best_params = None
    if 'inverse_config' not in st.session_state:
        st.session_state.inverse_config = None
    if 'inverse_study' not in st.session_state:
        st.session_state.inverse_study = None
    
    main()
