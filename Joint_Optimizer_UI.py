from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import streamlit as st

import Inverse_Optimizer_V5 as optimizer


APP_DIR = Path(__file__).resolve().parent
PARETO_PATH = APP_DIR / "joint_inverse_pareto_candidates.csv"
HOLDOUT_PATH = APP_DIR / "joint_holdout_validation_results.csv"
LOOCV_PATH = APP_DIR / "joint_train_loocv_metrics.csv"

st.set_page_config(
    page_title="Haynes 282 | Process Design",
    page_icon="H282",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #1d2b36;
        --navy: #173b5e;
        --teal: #3d8791;
        --gold: #b88936;
        --line: #d7e0e7;
        --muted: #596b78;
        --wash: #f2f6f8;
    }
    .stApp { background: #fbfcfd; color: var(--ink); }
    [data-testid="stSidebar"] { background: #eef3f6; border-right: 1px solid var(--line); }
    [data-testid="stHeader"] { background: rgba(251,252,253,0.96); }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1440px; }
    h1, h2, h3 { color: var(--navy); letter-spacing: 0; }
    h1 { font-size: 2rem; line-height: 1.15; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-top: 3px solid var(--teal); padding: 0.8rem 1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stTabs"] button { color: var(--navy); }
    .status-line { color: var(--muted); font-size: 0.9rem; border-bottom: 1px solid var(--line); padding-bottom: 0.8rem; }
    .notice { border-left: 4px solid var(--gold); padding: 0.7rem 0.9rem; background: #f8f4eb; color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_dataframe(frame, **kwargs):
    width_parameter = inspect.signature(st.dataframe).parameters.get("width")
    if width_parameter and width_parameter.default in ("stretch", "content"):
        st.dataframe(frame, width="stretch", **kwargs)
    else:
        st.dataframe(frame, use_container_width=True, **kwargs)

st.title("Haynes 282 | Process Design")
st.markdown(
    "Forward property prediction and inverse process-parameter search",
    help="Uses the joint hardness, UTS, and relative-density surrogate models trained from measured settings with synthetic augmentation.",
)
st.markdown(
    "**Joint model** · Hardness (HV) · UTS (MPa) · Relative density (%) "
    f"· Layer thickness fixed at {optimizer.FIXED_LAYER_THICKNESS:.3f} mm",
    help="Models use measured experimental settings with synthetic augmentation for process-domain coverage.",
)
st.markdown(
    '<div class="status-line">Predictions are surrogate estimates. The supported search region is based on measured process settings; confirm recommendations by manufacturing and testing.</div>',
    unsafe_allow_html=True,
)


def energy_density(power, speed, hatch, thickness):
    return power / (speed * hatch * thickness)


def property_scale():
    spans = (
        optimizer.conditions[optimizer.TARGETS].max()
        - optimizer.conditions[optimizer.TARGETS].min()
    ).replace(0, 1.0)
    return spans.to_dict()


def prediction_frame(parameters, predictions):
    rows = []
    for key, label, unit in (
        ("Hardness_HV", "Hardness", "HV"),
        ("UTS_MPa", "UTS", "MPa"),
        ("RelativeDensity_pct", "Relative density", "%"),
    ):
        rows.append({"Property": label, "Prediction": predictions[key], "Unit": unit})
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Process domain")
    bounds = optimizer.PARAM_BOUNDS
    st.caption(
        f"Power {bounds['LaserPower_W'][0]:.0f}-{bounds['LaserPower_W'][1]:.0f} W\n\n"
        f"Speed {bounds['ScanSpeed_mm_s'][0]:.0f}-{bounds['ScanSpeed_mm_s'][1]:.0f} mm/s\n\n"
        f"Hatch {bounds['HatchDistance_mm'][0]:.3f}-{bounds['HatchDistance_mm'][1]:.3f} mm"
    )
    st.metric("Fixed layer thickness", f"{optimizer.FIXED_LAYER_THICKNESS:.3f} mm")
    st.metric(
        "Observed VED window",
        f"{optimizer.ENERGY_DENSITY_BOUNDS[0]:.1f}-{optimizer.ENERGY_DENSITY_BOUNDS[1]:.1f}",
        help="J/mm³",
    )
    st.caption("Search domain and validation are based on measured experimental settings.")

forward_tab, inverse_tab, candidates_tab, validation_tab = st.tabs(
    ["Forward prediction", "Inverse targets", "Pareto candidates", "Validation"]
)

with forward_tab:
    st.subheader("Predict properties from process settings")
    st.caption("Enter settings within the measured bounds. Hatch and VED may still place a combination outside the observed experimental region.")
    p_col, v_col, h_col, t_col = st.columns(4)
    with p_col:
        power = st.number_input(
            "Laser power (W)",
            min_value=float(bounds["LaserPower_W"][0]),
            max_value=float(bounds["LaserPower_W"][1]),
            value=270.0,
            step=1.0,
        )
    with v_col:
        speed = st.number_input(
            "Scan speed (mm/s)",
            min_value=float(bounds["ScanSpeed_mm_s"][0]),
            max_value=float(bounds["ScanSpeed_mm_s"][1]),
            value=1500.0,
            step=10.0,
        )
    with h_col:
        hatch = st.number_input(
            "Hatch distance (mm)",
            min_value=float(bounds["HatchDistance_mm"][0]),
            max_value=float(bounds["HatchDistance_mm"][1]),
            value=0.080,
            step=0.001,
            format="%.3f",
        )
    with t_col:
        st.number_input(
            "Layer thickness (mm)",
            min_value=optimizer.FIXED_LAYER_THICKNESS,
            max_value=optimizer.FIXED_LAYER_THICKNESS,
            value=optimizer.FIXED_LAYER_THICKNESS,
            disabled=True,
            format="%.3f",
        )
    thickness = optimizer.FIXED_LAYER_THICKNESS
    parameters = {
        "LaserPower_W": power,
        "ScanSpeed_mm_s": speed,
        "HatchDistance_mm": hatch,
        "LayerThickness_mm": thickness,
    }
    ved = energy_density(power, speed, hatch, thickness)
    ved_low, ved_high = optimizer.ENERGY_DENSITY_BOUNDS
    if ved_low <= ved <= ved_high:
        st.success(f"VED = {ved:.2f} J/mm³ · inside observed range")
    else:
        st.warning(f"VED = {ved:.2f} J/mm³ · outside observed range ({ved_low:.2f}-{ved_high:.2f} J/mm³); prediction is extrapolative.")
    predictions = optimizer.predict_properties(parameters)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Hardness", f"{predictions['Hardness_HV']:.2f} HV")
    metric_cols[1].metric("UTS", f"{predictions['UTS_MPa']:.1f} MPa")
    metric_cols[2].metric("Relative density", f"{predictions['RelativeDensity_pct']:.3f}%")
    with st.expander("Prediction details"):
        show_dataframe(prediction_frame(parameters, predictions), hide_index=True)
        st.json({**parameters, "VED_J_mm3": ved})

with inverse_tab:
    st.subheader("Find process settings for target properties")
    st.caption("The optimizer minimizes equal-weight normalized target error and enforces the measured VED window.")
    target_cols = st.columns(3)
    with target_cols[0]:
        target_hardness = st.number_input("Target hardness (HV)", min_value=0.0, value=325.0, step=1.0)
    with target_cols[1]:
        target_uts = st.number_input("Target UTS (MPa)", min_value=0.0, value=900.0, step=5.0)
    with target_cols[2]:
        target_density = st.number_input("Target relative density (%)", min_value=0.0, max_value=100.0, value=99.5, step=0.1, format="%.2f")
    trials = st.slider("Optimization trials", min_value=100, max_value=1500, value=600, step=100)
    targets = {
        "Hardness_HV": target_hardness,
        "UTS_MPa": target_uts,
        "RelativeDensity_pct": target_density,
    }
    if st.button("Find matching settings", type="primary", key="target_search"):
        with st.spinner("Searching the measured process domain…"):
            params, predicted, loss = optimizer.optimize_for_targets(
                target_values=targets,
                models=optimizer.MODELS,
                parameter_bounds=optimizer.PARAM_BOUNDS,
                energy_density_bounds=optimizer.ENERGY_DENSITY_BOUNDS,
                target_scales=property_scale(),
                seed=optimizer.RANDOM_SEED,
                n_trials=trials,
            )
        ved = energy_density(
            params["LaserPower_W"],
            params["ScanSpeed_mm_s"],
            params["HatchDistance_mm"],
            params["LayerThickness_mm"],
        )
        result = {
            "LaserPower_W": params["LaserPower_W"],
            "ScanSpeed_mm_s": params["ScanSpeed_mm_s"],
            "HatchDistance_mm": params["HatchDistance_mm"],
            "LayerThickness_mm": params["LayerThickness_mm"],
            "VED_J_mm3": ved,
            "NormalizedTargetLoss": loss,
        }
        for target in optimizer.TARGETS:
            actual = targets[target]
            error = predicted[target] - actual
            pct = optimizer.signed_percent_error(predicted[target], actual)
            result[f"Target_{target}"] = actual
            result[f"Predicted_{target}"] = predicted[target]
            result[f"Error_{target}"] = error
            result[f"ErrorPercent_{target}"] = pct
            result[f"AbsoluteErrorPercent_{target}"] = abs(pct)
        st.session_state["joint_target_result"] = result
    if "joint_target_result" in st.session_state:
        result = st.session_state["joint_target_result"]
        st.markdown("#### Recommended process setting")
        param_cols = st.columns(4)
        param_cols[0].metric("Power", f"{result['LaserPower_W']:.2f} W")
        param_cols[1].metric("Speed", f"{result['ScanSpeed_mm_s']:.1f} mm/s")
        param_cols[2].metric("Hatch", f"{result['HatchDistance_mm']:.4f} mm")
        param_cols[3].metric("VED", f"{result['VED_J_mm3']:.2f} J/mm³")
        rows = []
        for key, label, unit in (
            ("Hardness_HV", "Hardness", "HV"),
            ("UTS_MPa", "UTS", "MPa"),
            ("RelativeDensity_pct", "Relative density", "%"),
        ):
            rows.append({
                "Property": label,
                "Target": result[f"Target_{key}"],
                "Predicted": result[f"Predicted_{key}"],
                "Signed error": result[f"Error_{key}"],
                "Error (%)": result[f"ErrorPercent_{key}"],
                "Unit": unit,
            })
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download target-match result",
            pd.DataFrame([result]).to_csv(index=False).encode("utf-8"),
            file_name="joint_inverse_target_match.csv",
            mime="text/csv",
        )
        st.markdown('<div class="notice">A close property match may correspond to different process settings. Treat this as a candidate for experimental validation, not a certified recipe.</div>', unsafe_allow_html=True)

with candidates_tab:
    st.subheader("Pareto candidate set")
    st.caption("Each row is a non-dominated trade-off; no single row maximizes every property.")
    if PARETO_PATH.exists():
        pareto = pd.read_csv(PARETO_PATH)
        display_columns = [
            "Candidate", "LaserPower_W", "ScanSpeed_mm_s", "HatchDistance_mm",
            "LayerThickness_mm", "VED_J_mm3", "Hardness_HV", "UTS_MPa",
            "RelativeDensity_pct", "BalancedScore",
        ]
        show_dataframe(pareto[display_columns].round(4), hide_index=True)
        st.download_button(
            "Download Pareto candidates",
            pareto.to_csv(index=False).encode("utf-8"),
            file_name="joint_inverse_pareto_candidates.csv",
            mime="text/csv",
        )
    else:
        st.info("No saved Pareto candidate file was found. Run V5 with --mode maximize first.")

with validation_tab:
    st.subheader("Measured validation and CV diagnostics")
    left, right = st.columns(2)
    if HOLDOUT_PATH.exists():
        with left:
            st.markdown("#### Forward holdout")
            holdout = pd.read_csv(HOLDOUT_PATH)
            show_dataframe(
                holdout[["LaserPower_W", "ScanSpeed_mm_s", "HatchDistance_mm", "Property", "Actual_Mean", "Predicted", "ErrorPercent"]].round(4),
                hide_index=True,
            )
    if LOOCV_PATH.exists():
        with right:
            st.markdown("#### Training-only LOOCV")
            cv = pd.read_csv(LOOCV_PATH)
            show_dataframe(cv.round(4), hide_index=True)
    st.warning("Validation diagnostics are preliminary and based on grouped experimental settings. Synthetic augmentation points are not independent experimental observations.")

st.caption("Haynes 282 LPBF · Joint surrogate models · Experimental verification required")
