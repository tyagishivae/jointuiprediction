import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna
from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE / CUSTOM CSS
# ============================================================

st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }

    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
    }

    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }

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

    .sidebar-section-title {
        color: #ffffff !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.8px;
        margin-top: 10px;
        margin-bottom: 10px;
        opacity: 1 !important;
    }

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

    .section-header {
        color: #003366;
        font-size: 1.4em;
        font-weight: 700;
        border-bottom: 3px solid #003366;
        padding-bottom: 10px;
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .target-box {
        background: #f7f9fc;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #d9e2ec;
        margin-bottom: 15px;
    }

    .success-box {
        background: #f0f9f6;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 18px;
        margin: 15px 0;
    }

    .warning-box {
        background: #fff9f0;
        border: 2px solid #fd7e14;
        border-radius: 10px;
        padding: 18px;
        margin: 15px 0;
    }

    .stMarkdown {
        color: #333333 !important;
    }

    div[data-testid="stMetric"] label {
        color: #003366 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #111111 !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #28a745 !important;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_DIR = Path("models")


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    """
    Load all trained models and feature order.
    """

    try:
        models = {
            "defect_classifier":
                joblib.load(MODEL_DIR / "defect_classifier.pkl"),

            "YieldStrength_MPa":
                joblib.load(MODEL_DIR / "YieldStrength_MPa.pkl"),

            "UTS_MPa":
                joblib.load(MODEL_DIR / "UTS_MPa.pkl"),

            "Hardness_HRA":
                joblib.load(MODEL_DIR / "Hardness_HRA.pkl"),

            "MeltPoolDepth_um":
                joblib.load(MODEL_DIR / "MeltPoolDepth_um.pkl"),

            "MeltPoolWidth_um":
                joblib.load(MODEL_DIR / "MeltPoolWidth_um.pkl"),

            "RelativeDensity_frac":
                joblib.load(MODEL_DIR / "RelativeDensity_frac.pkl"),
        }

        feature_order = joblib.load(
            MODEL_DIR / "feature_order.pkl"
        )

        return models, feature_order

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


# ============================================================
# INPUT CREATION
# ============================================================

def make_input(
    P_W,
    v_mm_per_s,
    h_mm,
    t_mm,
    feature_order
):
    """
    Create input dataframe in exact model feature order.
    """

    values = {
        "P_W": P_W,
        "v_mm_per_s": v_mm_per_s,
        "h_mm": h_mm,
        "t_mm": t_mm
    }

    row = {
        col: values[col]
        for col in feature_order
    }

    return pd.DataFrame([row])


# ============================================================
# FORWARD PREDICTION
# ============================================================

def predict_properties(
    P_W,
    v_mm_per_s,
    h_mm,
    t_mm,
    models,
    feature_order
):
    """
    Predict all available properties for one parameter set.
    """

    X = make_input(
        P_W,
        v_mm_per_s,
        h_mm,
        t_mm,
        feature_order
    )

    # --------------------------------------------------------
    # Defect prediction
    # --------------------------------------------------------

    try:
        probs = models["defect_classifier"].predict_proba(X)[0]

        defect_risk = 1 - np.max(probs)
        defect_class = int(np.argmax(probs))

    except Exception:

        defect_pred = int(
            models["defect_classifier"].predict(X)[0]
        )

        defect_class = defect_pred

        defect_risk = (
            0
            if defect_pred == 1
            else 0.5
        )

    # --------------------------------------------------------
    # Property predictions
    # --------------------------------------------------------

    uts = float(
        models["UTS_MPa"].predict(X)[0]
    )

    yield_strength = float(
        models["YieldStrength_MPa"].predict(X)[0]
    )

    hardness = float(
        models["Hardness_HRA"].predict(X)[0]
    )

    density = float(
        models["RelativeDensity_frac"].predict(X)[0]
    ) * 100

    width = float(
        models["MeltPoolWidth_um"].predict(X)[0]
    )

    depth = float(
        models["MeltPoolDepth_um"].predict(X)[0]
    )

    return {
        "UTS_MPa": uts,
        "YieldStrength_MPa": yield_strength,
        "Hardness_HRA": hardness,
        "RelativeDensity_pct": density,
        "MeltPoolWidth_um": width,
        "MeltPoolDepth_um": depth,
        "DefectClass": defect_class,
        "DefectRisk": defect_risk
    }


# ============================================================
# INVERSE OPTIMIZATION OBJECTIVE
# ============================================================

def objective(
    trial,
    models,
    feature_order,
    config
):
    """
    Optuna objective.

    Primary optimization:
        1. Target UTS
        2. Target Relative Density

    Secondary constraints:
        3. Defect regime
        4. Melt-pool width
        5. Melt-pool depth

    The optimizer searches for ONE common
    P, v, h, t combination.
    """

    # --------------------------------------------------------
    # Process parameters suggested by Optuna
    # --------------------------------------------------------

    P_W = trial.suggest_float(
        "P_W",
        config["P_min"],
        config["P_max"]
    )

    v_mm_per_s = trial.suggest_float(
        "v_mm_per_s",
        config["v_min"],
        config["v_max"]
    )

    h_mm = trial.suggest_float(
        "h_mm",
        config["h_min"],
        config["h_max"]
    )

    t_mm = trial.suggest_float(
        "t_mm",
        config["t_min"],
        config["t_max"]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        results = predict_properties(
            P_W,
            v_mm_per_s,
            h_mm,
            t_mm,
            models,
            feature_order
        )

        uts = results["UTS_MPa"]

        density = results["RelativeDensity_pct"]

        defect_class = results["DefectClass"]

        width = results["MeltPoolWidth_um"]

        depth = results["MeltPoolDepth_um"]

        # ----------------------------------------------------
        # NORMALIZED PRIMARY ERRORS
        # ----------------------------------------------------

        # UTS error
        uts_error = abs(
            uts - config["target_uts"]
        ) / config["uts_scale"]

        # Relative density error
        density_error = abs(
            density - config["target_density"]
        ) / config["density_scale"]

        # ----------------------------------------------------
        # PRIMARY OBJECTIVE
        # ----------------------------------------------------

        score = (
            config["w_uts"] * uts_error
            +
            config["w_density"] * density_error
        )

        # ----------------------------------------------------
        # DEFECT CONSTRAINT
        # ----------------------------------------------------

        if config["require_stable_defect"]:

            if defect_class != 1:
                score += config["defect_penalty"]

        # ----------------------------------------------------
        # MELT-POOL WIDTH CONSTRAINT
        # ----------------------------------------------------

        if width < config["min_width"]:

            score += (
                config["width_penalty"]
                *
                (
                    config["min_width"]
                    - width
                )
                /
                config["width_scale"]
            )

        elif width > config["max_width"]:

            score += (
                config["width_penalty"]
                *
                (
                    width
                    - config["max_width"]
                )
                /
                config["width_scale"]
            )

        # ----------------------------------------------------
        # MELT-POOL DEPTH CONSTRAINT
        # ----------------------------------------------------

        if depth < config["min_depth"]:

            score += (
                config["depth_penalty"]
                *
                (
                    config["min_depth"]
                    - depth
                )
                /
                config["depth_scale"]
            )

        elif depth > config["max_depth"]:

            score += (
                config["depth_penalty"]
                *
                (
                    depth
                    - config["max_depth"]
                )
                /
                config["depth_scale"]
            )

        return score

    except Exception:

        return float("inf")


# ============================================================
# TARGET MATCH CALCULATIONS
# ============================================================

def calculate_target_errors(
    results,
    config
):
    """
    Calculate absolute and percentage-style
    target deviations.
    """

    uts_error = (
        results["UTS_MPa"]
        -
        config["target_uts"]
    )

    density_error = (
        results["RelativeDensity_pct"]
        -
        config["target_density"]
    )

    uts_abs_error = abs(uts_error)

    density_abs_error = abs(density_error)

    uts_relative_error = (
        uts_abs_error
        /
        abs(config["target_uts"])
    ) * 100

    density_relative_error = (
        density_abs_error
        /
        abs(config["target_density"])
    ) * 100

    return {
        "UTS_Error_MPa": uts_error,
        "UTS_Absolute_Error_MPa": uts_abs_error,
        "UTS_Relative_Error_pct": uts_relative_error,

        "Density_Error_pct_points": density_error,
        "Density_Absolute_Error_pct_points": density_abs_error,
        "Density_Relative_Error_pct": density_relative_error
    }


# ============================================================
# PARAMETER EVALUATION
# ============================================================

def evaluate_parameters(
    P_W,
    v_mm_per_s,
    h_mm,
    t_mm,
    models,
    feature_order
):
    """
    Evaluate one parameter combination.
    """

    return predict_properties(
        P_W,
        v_mm_per_s,
        h_mm,
        t_mm,
        models,
        feature_order
    )


# ============================================================
# PROCESS CONSTRAINT CHECK
# ============================================================

def check_constraints(
    results,
    config
):
    """
    Check secondary process constraints.
    """

    stable_defect = (
        results["DefectClass"] == 1
    )

    width_ok = (
        config["min_width"]
        <=
        results["MeltPoolWidth_um"]
        <=
        config["max_width"]
    )

    depth_ok = (
        config["min_depth"]
        <=
        results["MeltPoolDepth_um"]
        <=
        config["max_depth"]
    )

    return {
        "stable_defect": stable_defect,
        "width_ok": width_ok,
        "depth_ok": depth_ok,
        "all_constraints_ok":
            (
                stable_defect
                and
                width_ok
                and
                depth_ok
            )
    }


# ============================================================
# METRIC CARD
# ============================================================

def show_metric_card(
    label,
    value,
    unit=""
):
    """
    Display a metric card.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value:.2f} {unit}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        '<div class="title-main">'
        'INVERSE OPTIMIZER'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Multi-Property LPBF Parameter Design '
        'for Target UTS and Relative Density'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models, feature_order = load_models()

    if (
        models is None
        or
        feature_order is None
    ):

        st.error(
            "Failed to load models. "
            "Please check the Models folder."
        )

        return

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.markdown(
        """
        <div class="sidebar-section-title">
            TARGET PROPERTIES
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("---")

    # --------------------------------------------------------
    # Target UTS
    # --------------------------------------------------------

    target_uts = st.sidebar.slider(
        "Target UTS (MPa)",
        min_value=700,
        max_value=1200,
        value=900,
        step=10,
        help=(
            "Desired Ultimate Tensile Strength "
            "for the printed part."
        )
    )

    # --------------------------------------------------------
    # Target density
    # --------------------------------------------------------

    target_density = st.sidebar.slider(
        "Target Relative Density (%)",
        min_value=95.0,
        max_value=100.0,
        value=99.5,
        step=0.01,
        help=(
            "Desired relative density "
            "for the printed part."
        )
    )

    # ========================================================
    # OBJECTIVE WEIGHTS
    # ========================================================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        <div class="sidebar-section-title">
            OBJECTIVE WEIGHTS
        </div>
        """,
        unsafe_allow_html=True
    )

    uts_weight_percent = st.sidebar.slider(
        "UTS Importance (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5
    )

    density_weight_percent = (
        100 - uts_weight_percent
    )

    st.sidebar.metric(
        "Relative Density Importance",
        f"{density_weight_percent}%"
    )

    w_uts = (
        uts_weight_percent
        /
        100
    )

    w_density = (
        density_weight_percent
        /
        100
    )

    # ========================================================
    # PROCESS CONSTRAINTS
    # ========================================================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        <div class="sidebar-section-title">
            PROCESS CONSTRAINTS
        </div>
        """,
        unsafe_allow_html=True
    )

    require_stable_defect = st.sidebar.checkbox(
        "Require Stable Defect Regime",
        value=True,
        help=(
            "Penalize candidate parameter sets "
            "predicted outside the stable regime."
        )
    )

    # --------------------------------------------------------
    # Melt-pool width
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Melt-pool depth
    # --------------------------------------------------------

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

    # ========================================================
    # OPTIMIZATION SETTINGS
    # ========================================================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        <div class="sidebar-section-title">
            OPTIMIZATION SETTINGS
        </div>
        """,
        unsafe_allow_html=True
    )

    n_trials = st.sidebar.slider(
        "Optimization Trials",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help=(
            "Number of candidate process "
            "parameter combinations evaluated."
        )
    )

    # --------------------------------------------------------
    # Run button
    # --------------------------------------------------------

    st.sidebar.markdown("---")

    optimize_clicked = st.sidebar.button(
        "RUN OPTIMIZATION",
        use_container_width=True
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    config = {

        # Target properties
        "target_uts":
            target_uts,

        "target_density":
            target_density,

        # Objective weights
        "w_uts":
            w_uts,

        "w_density":
            w_density,

        # Normalization scales
        #
        # These are optimization scaling factors,
        # not physical constants.
        "uts_scale":
            100.0,

        "density_scale":
            1.0,

        # Defect constraint
        "require_stable_defect":
            require_stable_defect,

        "defect_penalty":
            5.0,

        # Melt-pool constraints
        "min_width":
            min_width,

        "max_width":
            max_width,

        "min_depth":
            min_depth,

        "max_depth":
            max_depth,

        # Penalty scaling
        "width_penalty":
            1.0,

        "depth_penalty":
            1.0,

        "width_scale":
            50.0,

        "depth_scale":
            50.0,

        # ----------------------------------------------------
        # LPBF PROCESS PARAMETER SEARCH SPACE
        # ----------------------------------------------------

        "P_min":
            150,

        "P_max":
            370,

        "v_min":
            350,

        "v_max":
            1772,

        "h_min":
            0.09,

        "h_max":
            0.12,

        "t_min":
            0.03,

        "t_max":
            0.05,
    }

    # ========================================================
    # RUN OPTIMIZATION
    # ========================================================

    if optimize_clicked:

        with st.spinner(
            "Running multi-property inverse optimization..."
        ):

            progress_bar = st.progress(0)

            study = optuna.create_study(
                direction="minimize"
            )

            # ------------------------------------------------
            # Progress callback
            # ------------------------------------------------

            def callback(
                study,
                trial
            ):

                progress = (
                    (trial.number + 1)
                    /
                    n_trials
                )

                progress_bar.progress(
                    min(progress, 1.0)
                )

            # ------------------------------------------------
            # Run Optuna
            # ------------------------------------------------

            study.optimize(
                lambda trial:
                    objective(
                        trial,
                        models,
                        feature_order,
                        config
                    ),

                n_trials=n_trials,

                callbacks=[
                    callback
                ]
            )

        progress_bar.progress(1.0)

        # ----------------------------------------------------
        # Best parameters
        # ----------------------------------------------------

        best_params = study.best_params

        st.session_state.inverse_best_params = (
            best_params
        )

        st.session_state.inverse_config = (
            config
        )

        st.session_state.inverse_show_results = (
            True
        )

        st.session_state.inverse_study = (
            study
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if (
        st.session_state.get(
            "inverse_show_results",
            False
        )
        and
        st.session_state.get(
            "inverse_best_params"
        ) is not None
    ):

        best_params = (
            st.session_state.inverse_best_params
        )

        config = (
            st.session_state.inverse_config
        )

        # ----------------------------------------------------
        # Evaluate optimal parameter combination
        # ----------------------------------------------------

        results = evaluate_parameters(
            best_params["P_W"],
            best_params["v_mm_per_s"],
            best_params["h_mm"],
            best_params["t_mm"],
            models,
            feature_order
        )

        # ----------------------------------------------------
        # Calculate target errors
        # ----------------------------------------------------

        errors = calculate_target_errors(
            results,
            config
        )

        # ----------------------------------------------------
        # Check constraints
        # ----------------------------------------------------

        constraints = check_constraints(
            results,
            config
        )

        # ====================================================
        # SECTION 1
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            'RECOMMENDED PROCESS PARAMETERS'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            show_metric_card(
                "Laser Power",
                best_params["P_W"],
                "W"
            )

        with col2:

            show_metric_card(
                "Scan Speed",
                best_params["v_mm_per_s"],
                "mm/s"
            )

        with col3:

            show_metric_card(
                "Hatch Spacing",
                best_params["h_mm"],
                "mm"
            )

        with col4:

            show_metric_card(
                "Layer Thickness",
                best_params["t_mm"],
                "mm"
            )

        # ====================================================
        # SECTION 2
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            'TARGET VS PREDICTED PERFORMANCE'
            '</div>',
            unsafe_allow_html=True
        )

        comparison_df = pd.DataFrame({

            "Property": [
                "Relative Density",
                "UTS"
            ],

            "Target": [
                target_density,
                target_uts
            ],

            "Predicted": [
                results[
                    "RelativeDensity_pct"
                ],
                results[
                    "UTS_MPa"
                ]
            ],

            "Absolute Error": [
                errors[
                    "Density_Absolute_Error_pct_points"
                ],
                errors[
                    "UTS_Absolute_Error_MPa"
                ]
            ],

            "Unit": [
                "percentage points",
                "MPa"
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # SECTION 3
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            'PREDICTED MATERIAL PERFORMANCE'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            show_metric_card(
                "Relative Density",
                results[
                    "RelativeDensity_pct"
                ],
                "%"
            )

        with col2:

            show_metric_card(
                "UTS",
                results["UTS_MPa"],
                "MPa"
            )

        with col3:

            show_metric_card(
                "Yield Strength",
                results["YieldStrength_MPa"],
                "MPa"
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            show_metric_card(
                "Hardness",
                results["Hardness_HRA"],
                "HRA"
            )

        with col2:

            show_metric_card(
                "Meltpool Width",
                results["MeltPoolWidth_um"],
                "µm"
            )

        with col3:

            show_metric_card(
                "Meltpool Depth",
                results["MeltPoolDepth_um"],
                "µm"
            )

        # ====================================================
        # SECTION 4
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            'PROCESS CONSTRAINT STATUS'
            '</div>',
            unsafe_allow_html=True
        )

        defect_mapping = {
            0: "Lack of Fusion",
            1: "Stable",
            2: "Keyhole"
        }

        defect_status = defect_mapping.get(
            results["DefectClass"],
            "Unknown"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            if constraints["stable_defect"]:

                st.success(
                    f"Defect Regime: {defect_status}"
                )

            else:

                st.warning(
                    f"Defect Regime: {defect_status}"
                )

        with c2:

            if constraints["width_ok"]:

                st.success(
                    "Meltpool Width: Within Range"
                )

            else:

                st.warning(
                    "Meltpool Width: Outside Range"
                )

        with c3:

            if constraints["depth_ok"]:

                st.success(
                    "Meltpool Depth: Within Range"
                )

            else:

                st.warning(
                    "Meltpool Depth: Outside Range"
                )

        # ====================================================
        # SECTION 5
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            'TARGET MATCH SUMMARY'
            '</div>',
            unsafe_allow_html=True
        )

        target_match = (
            errors["UTS_Absolute_Error_MPa"]
            <=
            config["uts_scale"] * 0.05
            and
            errors[
                "Density_Absolute_Error_pct_points"
            ]
            <=
            config["density_scale"] * 0.05
        )

        all_constraints_ok = (
            constraints["all_constraints_ok"]
        )

        if (
            target_match
            and
            all_constraints_ok
        ):

            st.markdown(
                """
                <div class="success-box">
                    <h3>Target-Compatible Parameter Set</h3>

                    <p>
                    The optimized process parameters provide
                    predicted UTS and relative density close
                    to the requested targets while satisfying
                    the configured process constraints.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="warning-box">
                    <h3>Closest Feasible Solution Found</h3>

                    <p>
                    The optimizer returned the best available
                    parameter combination within the defined
                    process search space. One or more targets
                    or secondary constraints may not be fully
                    satisfied.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # SECTION 6
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            'OPTIMIZATION DETAILS'
            '</div>',
            unsafe_allow_html=True
        )

        study = (
            st.session_state.inverse_study
        )

        trials_df = (
            study.trials_dataframe()
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Trials",
                len(trials_df)
            )

        with col2:

            st.metric(
                "Best Trial",
                study.best_trial.number
            )

        with col3:

            st.metric(
                "Best Objective",
                f"{study.best_value:.6f}"
            )

        # ----------------------------------------------------
        # Optimization history
        # ----------------------------------------------------

        st.markdown(
            "### Optimization Trial History"
        )

        available_cols = [
            "number",
            "value",
            "state"
        ]

        for col in [
            "P_W",
            "v_mm_per_s",
            "h_mm",
            "t_mm"
        ]:

            if col in trials_df.columns:

                available_cols.append(
                    col
                )

        st.dataframe(
            trials_df[
                available_cols
            ].tail(10),

            use_container_width=True
        )

        # ====================================================
        # SECTION 7
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            'EXPORT RESULTS'
            '</div>',
            unsafe_allow_html=True
        )

        export_data = {

            "Timestamp": [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ],

            # -----------------------------------------------
            # Targets
            # -----------------------------------------------

            "Target_UTS_MPa": [
                target_uts
            ],

            "Target_RelativeDensity_pct": [
                target_density
            ],

            # -----------------------------------------------
            # Optimized parameters
            # -----------------------------------------------

            "Power_W": [
                best_params["P_W"]
            ],

            "ScanSpeed_mm_per_s": [
                best_params["v_mm_per_s"]
            ],

            "HatchSpacing_mm": [
                best_params["h_mm"]
            ],

            "LayerThickness_mm": [
                best_params["t_mm"]
            ],

            # -----------------------------------------------
            # Predictions
            # -----------------------------------------------

            "Predicted_UTS_MPa": [
                results["UTS_MPa"]
            ],

            "Predicted_RelativeDensity_pct": [
                results["RelativeDensity_pct"]
            ],

            "Predicted_YieldStrength_MPa": [
                results["YieldStrength_MPa"]
            ],

            "Predicted_Hardness_HRA": [
                results["Hardness_HRA"]
            ],

            "Predicted_MeltPoolWidth_um": [
                results["MeltPoolWidth_um"]
            ],

            "Predicted_MeltPoolDepth_um": [
                results["MeltPoolDepth_um"]
            ],

            # -----------------------------------------------
            # Errors
            # -----------------------------------------------

            "UTS_Error_MPa": [
                errors["UTS_Error_MPa"]
            ],

            "UTS_Absolute_Error_MPa": [
                errors[
                    "UTS_Absolute_Error_MPa"
                ]
            ],

            "UTS_Relative_Error_pct": [
                errors[
                    "UTS_Relative_Error_pct"
                ]
            ],

            "Density_Error_percentage_points": [
                errors[
                    "Density_Error_pct_points"
                ]
            ],

            "Density_Absolute_Error_percentage_points": [
                errors[
                    "Density_Absolute_Error_pct_points"
                ]
            ],

            "Density_Relative_Error_pct": [
                errors[
                    "Density_Relative_Error_pct"
                ]
            ],

            # -----------------------------------------------
            # Process information
            # -----------------------------------------------

            "DefectClass": [
                defect_status
            ],

            "DefectRisk": [
                results["DefectRisk"]
            ],

            "StableDefectConstraint": [
                constraints["stable_defect"]
            ],

            "WidthConstraint": [
                constraints["width_ok"]
            ],

            "DepthConstraint": [
                constraints["depth_ok"]
            ],

            "AllConstraintsSatisfied": [
                constraints[
                    "all_constraints_ok"
                ]
            ],

            # -----------------------------------------------
            # Objective information
            # -----------------------------------------------

            "UTS_Weight": [
                w_uts
            ],

            "Density_Weight": [
                w_density
            ],

            "OptimizationObjective": [
                study.best_value
            ]
        }

        export_df = pd.DataFrame(
            export_data
        )

        csv_data = (
            export_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=(
                "Download Optimization "
                "Report (CSV)"
            ),

            data=csv_data,

            file_name=(
                "HAYNES282_"
                "MultiProperty_InverseOptimization_"
                +
                datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                +
                ".csv"
            ),

            mime="text/csv",

            use_container_width=True
        )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    else:

        col1, col2, col3 = st.columns(
            [1, 2, 1]
        )

        with col2:

            st.markdown(
                """
                ### Multi-Property Inverse Optimizer

                This tool searches for **one common LPBF
                process-parameter combination** that
                simultaneously approaches the specified
                Relative Density and UTS targets.

                **How it works:**

                1. Define target Relative Density and UTS.
                2. Set their relative optimization importance.
                3. Configure acceptable process constraints.
                4. Optuna searches the allowed LPBF
                   parameter space.
                5. Each candidate is evaluated using the
                   trained forward ML models.
                6. The optimizer minimizes the combined
                   normalized error between predicted and
                   target properties.
                7. The final parameter combination is
                   evaluated against defect and melt-pool
                   constraints.

                **Primary optimization targets:**

                - Relative Density
                - Ultimate Tensile Strength

                **Process parameters optimized:**

                - Laser Power
                - Scan Speed
                - Hatch Spacing
                - Layer Thickness

                **Additional predictions:**

                - Yield Strength
                - Hardness
                - Melt-pool Width
                - Melt-pool Depth
                - Defect Regime

                ---

                **Set your targets in the sidebar and
                click RUN OPTIMIZATION.**
                """
            )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if "inverse_show_results" not in st.session_state:

        st.session_state.inverse_show_results = False

    if "inverse_best_params" not in st.session_state:

        st.session_state.inverse_best_params = None

    if "inverse_config" not in st.session_state:

        st.session_state.inverse_config = None

    if "inverse_study" not in st.session_state:

        st.session_state.inverse_study = None

    main()