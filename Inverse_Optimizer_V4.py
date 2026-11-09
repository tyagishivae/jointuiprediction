# # ============================================================
# # Inverse_Optimizer_V3.py
# #
# # Joint inverse optimization for:
# #   1. Ultimate Tensile Strength (UTS)
# #   2. Relative Density
# #
# # Fixed layer thickness: 40 microns = 0.04 mm
# #
# # Optimized parameters:
# #   P = Laser Power (W)
# #   v = Scan Speed (mm/s)
# #   h = Hatch Spacing (mm)
# #
# # Models:
# #   models/UTS_MPa.pkl
# #   models/RelativeDensity_frac.pkl
# #
# # Feature schemas:
# #   models/UTS_feature_order.pkl
# #   models/feature_order.pkl
# # ============================================================

# import os
# import sys
# import joblib
# import numpy as np
# import pandas as pd
# import optuna


# # ============================================================
# # 1. CONFIGURATION
# # ============================================================

# MODEL_DIR = "models"

# UTS_MODEL_PATH = os.path.join(MODEL_DIR, "UTS_MPa.pkl")
# DENSITY_MODEL_PATH = os.path.join(
#     MODEL_DIR, "RelativeDensity_frac.pkl"
# )

# UTS_FEATURE_ORDER_PATH = os.path.join(
#     MODEL_DIR, "UTS_feature_order.pkl"
# )
# DENSITY_FEATURE_ORDER_PATH = os.path.join(
#     MODEL_DIR, "feature_order.pkl"
# )

# # Optional models: loaded only if present
# MELT_POOL_WIDTH_PATH = os.path.join(
#     MODEL_DIR, "MeltPoolWidth_um.pkl"
# )
# MELT_POOL_DEPTH_PATH = os.path.join(
#     MODEL_DIR, "MeltPoolDepth_um.pkl"
# )
# DEFECT_MODEL_PATH = os.path.join(
#     MODEL_DIR, "defect_classifier.pkl"
# )

# # Fixed layer thickness: 40 microns
# FIXED_LAYER_THICKNESS = 0.04  # mm

# # Search bounds
# # Adjust these bounds if your validated training domain differs.
# PARAM_BOUNDS = {
#     "P_W": (150.0, 370.0),
#     "v_mm_per_s": (350.0, 1772.0),
#     "h_mm": (0.09, 0.12),
# }

# # Optimization settings
# N_TRIALS = 1000
# RANDOM_SEED = 42

# # Target properties
# TARGET_UTS_MPA = 900.0
# TARGET_DENSITY_PERCENT = 99.5

# # Equal importance for UTS and relative density
# UTS_WEIGHT = 0.5
# DENSITY_WEIGHT = 0.5


# # ============================================================
# # 2. LOAD MODEL FEATURE SCHEMAS
# # ============================================================

# def extract_feature_names(feature_order):
#     """
#     Extract feature names from common serialized schema formats.
#     """
#     if isinstance(feature_order, dict):
#         for key in (
#             "feature_names",
#             "features",
#             "feature_order",
#             "columns",
#         ):
#             if key in feature_order:
#                 return extract_feature_names(feature_order[key])

#         raise ValueError(
#             "Could not identify feature names in the feature-order "
#             f"dictionary. Available keys: {list(feature_order.keys())}"
#         )

#     if isinstance(feature_order, (list, tuple, np.ndarray, pd.Index)):
#         return list(feature_order)

#     raise TypeError(
#         "Unsupported feature-order format: "
#         f"{type(feature_order).__name__}"
#     )


# def load_feature_order(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(
#             f"Feature-order file not found: {path}"
#         )

#     obj = joblib.load(path)
#     names = extract_feature_names(obj)

#     if not names:
#         raise ValueError(f"No features found in {path}")

#     return names


# # ============================================================
# # 3. LOAD MODELS
# # ============================================================

# def load_required_model(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(
#             f"Required model not found: {path}"
#         )

#     return joblib.load(path)


# print("\nLoading trained models...")

# uts_model = load_required_model(UTS_MODEL_PATH)
# density_model = load_required_model(DENSITY_MODEL_PATH)

# uts_features = load_feature_order(UTS_FEATURE_ORDER_PATH)
# density_features = load_feature_order(DENSITY_FEATURE_ORDER_PATH)

# print("UTS model loaded.")
# print("Relative-density model loaded.")
# print(f"UTS features: {uts_features}")
# print(f"Density features: {density_features}")

# # Optional models are for additional reporting only.
# melt_pool_width_model = (
#     joblib.load(MELT_POOL_WIDTH_PATH)
#     if os.path.exists(MELT_POOL_WIDTH_PATH)
#     else None
# )

# melt_pool_depth_model = (
#     joblib.load(MELT_POOL_DEPTH_PATH)
#     if os.path.exists(MELT_POOL_DEPTH_PATH)
#     else None
# )

# defect_model = (
#     joblib.load(DEFECT_MODEL_PATH)
#     if os.path.exists(DEFECT_MODEL_PATH)
#     else None
# )


# # ============================================================
# # 4. PROCESS PARAMETER INPUT
# # ============================================================

# def build_feature_row(feature_names, P, v, h, t):
#     """
#     Build one model input row using that model's own feature order.

#     Common aliases are supported. If your saved schema uses other
#     feature names, add the appropriate alias below.
#     """

#     aliases = {
#         "P_W": P,
#         "P": P,
#         "power": P,
#         "laser_power": P,
#         "laser_power_W": P,

#         "v_mm_per_s": v,
#         "v": v,
#         "speed": v,
#         "scan_speed": v,
#         "scan_speed_mm_s": v,

#         "h_mm": h,
#         "h": h,
#         "hatch": h,
#         "hatch_spacing": h,
#         "hatch_spacing_mm": h,

#         "t_mm": t,
#         "t": t,
#         "thickness": t,
#         "layer_thickness": t,
#         "layer_thickness_mm": t,
#     }

#     # Match feature names case-insensitively.
#     normalized_aliases = {
#         str(key).strip().lower(): value
#         for key, value in aliases.items()
#     }

#     row = {}

#     for feature in feature_names:
#         feature_key = str(feature).strip().lower()

#         if feature_key in normalized_aliases:
#             row[feature] = normalized_aliases[feature_key]
#         else:
#             raise ValueError(
#                 f"Cannot map required model feature '{feature}'. "
#                 "Update build_feature_row() to map this feature "
#                 "to the appropriate process parameter."
#             )

#     return pd.DataFrame([row], columns=feature_names)


# # ============================================================
# # 5. FORWARD PREDICTION FUNCTIONS
# # ============================================================

# def predict_scalar(model, feature_names, P, v, h, t):
#     X = build_feature_row(feature_names, P, v, h, t)

#     prediction = np.asarray(model.predict(X)).reshape(-1)

#     if len(prediction) == 0:
#         raise ValueError("The model returned an empty prediction.")

#     value = float(prediction[0])

#     if not np.isfinite(value):
#         raise ValueError(f"Model returned invalid prediction: {value}")

#     return value


# def predict_uts(P, v, h, t):
#     return predict_scalar(
#         uts_model, uts_features, P, v, h, t
#     )


# def predict_density_percent(P, v, h, t):
#     raw = predict_scalar(
#         density_model, density_features, P, v, h, t
#     )

#     # Handle either fractional density (e.g. 0.995)
#     # or percentage density (e.g. 99.5).
#     if 0.0 <= raw <= 1.0:
#         return raw * 100.0

#     return raw


# # ============================================================
# # 6. OBJECTIVE FUNCTION
# # ============================================================

# def objective(trial):
#     # Only three parameters are optimized.
#     P = trial.suggest_float(
#         "P_W",
#         PARAM_BOUNDS["P_W"][0],
#         PARAM_BOUNDS["P_W"][1],
#     )

#     v = trial.suggest_float(
#         "v_mm_per_s",
#         PARAM_BOUNDS["v_mm_per_s"][0],
#         PARAM_BOUNDS["v_mm_per_s"][1],
#     )

#     h = trial.suggest_float(
#         "h_mm",
#         PARAM_BOUNDS["h_mm"][0],
#         PARAM_BOUNDS["h_mm"][1],
#     )

#     # Layer thickness is fixed and never optimized.
#     t = FIXED_LAYER_THICKNESS

#     try:
#         predicted_uts = predict_uts(P, v, h, t)
#         predicted_density = predict_density_percent(P, v, h, t)

#         # Relative errors make the two targets comparable.
#         uts_relative_error = (
#             predicted_uts - TARGET_UTS_MPA
#         ) / abs(TARGET_UTS_MPA)

#         density_relative_error = (
#             predicted_density - TARGET_DENSITY_PERCENT
#         ) / abs(TARGET_DENSITY_PERCENT)

#         # Equal-weight joint objective.
#         loss = (
#             UTS_WEIGHT * uts_relative_error**2
#             + DENSITY_WEIGHT * density_relative_error**2
#         )

#         trial.set_user_attr("predicted_uts_mpa", predicted_uts)
#         trial.set_user_attr(
#             "predicted_density_percent", predicted_density
#         )
#         trial.set_user_attr("layer_thickness_mm", t)

#         if not np.isfinite(loss):
#             return float("inf")

#         return float(loss)

#     except Exception as exc:
#         trial.set_user_attr("prediction_error", str(exc))
#         return float("inf")


# # ============================================================
# # 7. RUN OPTIMIZATION
# # ============================================================

# def main():
#     global TARGET_UTS_MPA, TARGET_DENSITY_PERCENT

#     print("\n" + "=" * 65)
#     print("JOINT INVERSE OPTIMIZATION: UTS + RELATIVE DENSITY")
#     print("=" * 65)

#     print("\nEnter target properties.")

#     try:
#         value = input(
#             f"Target UTS in MPa [{TARGET_UTS_MPA}]: "
#         ).strip()

#         if value:
#             TARGET_UTS_MPA = float(value)

#         value = input(
#             "Target relative density in % "
#             f"[{TARGET_DENSITY_PERCENT}]: "
#         ).strip()

#         if value:
#             TARGET_DENSITY_PERCENT = float(value)

#     except ValueError:
#         print("Invalid input. Please enter numeric target values.")
#         sys.exit(1)

#     if TARGET_UTS_MPA <= 0 or TARGET_DENSITY_PERCENT <= 0:
#         print("Target UTS and density must be greater than zero.")
#         sys.exit(1)

#     print("\nOptimization configuration:")
#     print(f"Target UTS          : {TARGET_UTS_MPA:.4f} MPa")
#     print(
#         f"Target relative density: "
#         f"{TARGET_DENSITY_PERCENT:.4f}%"
#     )
#     print(
#         f"Fixed layer thickness: "
#         f"{FIXED_LAYER_THICKNESS:.4f} mm "
#         f"({FIXED_LAYER_THICKNESS * 1000:.0f} microns)"
#     )
#     print(f"Trials              : {N_TRIALS}")
#     print(f"Objective weights   : UTS={UTS_WEIGHT}, "
#           f"Density={DENSITY_WEIGHT}")

#     sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

#     study = optuna.create_study(
#         direction="minimize",
#         sampler=sampler,
#     )

#     study.optimize(
#         objective,
#         n_trials=N_TRIALS,
#         show_progress_bar=True,
#     )

#     # Exclude failed trials from the final result.
#     valid_trials = [
#         trial for trial in study.trials
#         if trial.state == optuna.trial.TrialState.COMPLETE
#         and np.isfinite(trial.value)
#     ]

#     if not valid_trials:
#         print(
#             "\nOptimization failed: no valid predictions were obtained."
#         )
#         print(
#             "Check the model feature schemas, parameter names, "
#             "and model input requirements."
#         )

#         failed_trials = [
#             trial for trial in study.trials
#             if trial.user_attrs.get("prediction_error")
#         ]

#         if failed_trials:
#             print("\nExample prediction error:")
#             print(failed_trials[0].user_attrs["prediction_error"])

#         sys.exit(1)

#     best_trial = min(valid_trials, key=lambda trial: trial.value)
#     best_params = best_trial.params

#     BEST_P = best_params["P_W"]
#     BEST_V = best_params["v_mm_per_s"]
#     BEST_H = best_params["h_mm"]
#     BEST_T = FIXED_LAYER_THICKNESS

#     # Recalculate predictions for the selected solution.
#     BEST_UTS = predict_uts(BEST_P, BEST_V, BEST_H, BEST_T)
#     BEST_DENSITY = predict_density_percent(
#         BEST_P, BEST_V, BEST_H, BEST_T
#     )

#     uts_error = BEST_UTS - TARGET_UTS_MPA
#     density_error = BEST_DENSITY - TARGET_DENSITY_PERCENT

#     uts_percentage_error = (
#         abs(uts_error) / abs(TARGET_UTS_MPA)
#     ) * 100.0

#     density_percentage_error = (
#         abs(density_error) / abs(TARGET_DENSITY_PERCENT)
#     ) * 100.0

#     # ========================================================
#     # 8. PRINT RESULTS
#     # ========================================================

#     print("\n" + "=" * 65)
#     print("OPTIMIZATION RESULTS")
#     print("=" * 65)

#     print("\nBest process parameters:")
#     print(f"Laser power (P)       : {BEST_P:.6f} W")
#     print(f"Scan speed (v)        : {BEST_V:.6f} mm/s")
#     print(f"Hatch spacing (h)     : {BEST_H:.6f} mm")
#     print(
#         f"Layer thickness (t)   : {BEST_T:.6f} mm "
#         f"({BEST_T * 1000:.0f} microns; FIXED)"
#     )

#     print("\nProperty predictions:")
#     print(
#         f"Target UTS            : {TARGET_UTS_MPA:.6f} MPa"
#     )
#     print(
#         f"Predicted UTS         : {BEST_UTS:.6f} MPa"
#     )
#     print(f"UTS signed error      : {uts_error:+.6f} MPa")
#     print(
#         f"UTS absolute % error  : {uts_percentage_error:.6f}%"
#     )

#     print(
#         f"\nTarget relative density   : "
#         f"{TARGET_DENSITY_PERCENT:.6f}%"
#     )
#     print(
#         f"Predicted relative density: {BEST_DENSITY:.6f}%"
#     )
#     print(
#         f"Density signed error      : "
#         f"{density_error:+.6f} percentage points"
#     )
#     print(
#         f"Density absolute % error  : "
#         f"{density_percentage_error:.6f}%"
#     )

#     print(f"\nBest objective value  : {best_trial.value:.10f}")
#     print(
#         f"Successful trials     : {len(valid_trials)} / {N_TRIALS}"
#     )

#     # ========================================================
#     # 9. OPTIONAL MODEL OUTPUTS
#     # ========================================================

#     print("\nOptional model predictions:")

#     if melt_pool_width_model is not None:
#         try:
#             # Uses the UTS schema as a default; change this if the
#             # melt-pool model was trained with a different schema.
#             width = predict_scalar(
#                 melt_pool_width_model,
#                 uts_features,
#                 BEST_P, BEST_V, BEST_H, BEST_T,
#             )
#             print(f"Melt pool width       : {width:.6f} (model units)")
#         except Exception as exc:
#             print(f"Melt pool width       : unavailable ({exc})")
#     else:
#         print("Melt pool width       : model not found")

#     if melt_pool_depth_model is not None:
#         try:
#             # Uses the UTS schema as a default; change this if the
#             # melt-pool model was trained with a different schema.
#             depth = predict_scalar(
#                 melt_pool_depth_model,
#                 uts_features,
#                 BEST_P, BEST_V, BEST_H, BEST_T,
#             )
#             print(f"Melt pool depth       : {depth:.6f} (model units)")
#         except Exception as exc:
#             print(f"Melt pool depth       : unavailable ({exc})")
#     else:
#         print("Melt pool depth       : model not found")

#     if defect_model is not None:
#         try:
#             # This assumes the defect classifier uses the density
#             # model's feature schema. Verify against its training schema.
#             X_defect = build_feature_row(
#                 density_features,
#                 BEST_P, BEST_V, BEST_H, BEST_T,
#             )

#             prediction = defect_model.predict(X_defect)
#             print(f"Defect classifier     : {prediction[0]}")
#         except Exception as exc:
#             print(f"Defect classifier     : unavailable ({exc})")
#     else:
#         print("Defect classifier     : model not found")

#     # ========================================================
#     # 10. SAVE RESULTS
#     # ========================================================

#     results = pd.DataFrame([{
#         "P_W": BEST_P,
#         "v_mm_per_s": BEST_V,
#         "h_mm": BEST_H,
#         "t_mm": BEST_T,
#         "target_UTS_MPa": TARGET_UTS_MPA,
#         "predicted_UTS_MPa": BEST_UTS,
#         "UTS_signed_error_MPa": uts_error,
#         "UTS_absolute_percentage_error": uts_percentage_error,
#         "target_density_percent": TARGET_DENSITY_PERCENT,
#         "predicted_density_percent": BEST_DENSITY,
#         "density_signed_error_percentage_points": density_error,
#         "density_absolute_percentage_error": density_percentage_error,
#         "objective_value": best_trial.value,
#         "successful_trials": len(valid_trials),
#     }])

#     output_path = "inverse_optimization_results.csv"
#     results.to_csv(output_path, index=False)

#     print(f"\nResults saved to: {output_path}")
#     print("\nNote: predictions are model estimates, not experimental")
#     print("validation of the optimized process parameters.")



# Dated - 13 October 26 , new version 

# if __name__ == "__main__":
#     main()
# ============================================================
# Inverse_Optimizer_V4.py
#
# LPBF inverse optimization with:
#   Mode 1: Match target UTS and relative density
#   Mode 2: Maximize UTS and relative density (Pareto solutions)
#
# Fixed layer thickness: 40 microns = 0.04 mm
# Energy-density constraint: 45 to 55 J/mm^3
#
# Required files:
#   models/UTS_MPa.pkl
#   models/RelativeDensity_frac.pkl
#   models/UTS_feature_order.pkl
#   models/feature_order.pkl
# ============================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import optuna


# ------------------------- CONFIGURATION -------------------------

MODEL_DIR = "models"

UTS_MODEL_PATH = os.path.join(MODEL_DIR, "UTS_MPa.pkl")
DENSITY_MODEL_PATH = os.path.join(MODEL_DIR, "RelativeDensity_frac.pkl")
UTS_FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "UTS_feature_order.pkl")
DENSITY_FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.pkl")

FIXED_LAYER_THICKNESS = 0.04  # mm = 40 microns

ENERGY_DENSITY_MIN = 45.0  # J/mm^3
ENERGY_DENSITY_MAX = 55.0  # J/mm^3

PARAM_BOUNDS = {
    "P_W": (150.0, 370.0),
    "v_mm_per_s": (350.0, 1772.0),
    "h_mm": (0.09, 0.12),
}

N_TRIALS = 300
RANDOM_SEED = 42

DEFAULT_TARGET_UTS_MPA = 900.0
DEFAULT_TARGET_DENSITY_PERCENT = 99.5
UTS_WEIGHT = 0.5
DENSITY_WEIGHT = 0.5

OUTPUT_TARGET_MODE = "inverse_optimization_results.csv"
OUTPUT_MAX_MODE = "maximum_properties_pareto_results.csv"


# ------------------------- LOAD SCHEMAS -------------------------

def extract_feature_names(feature_order):
    """Extract feature names from common serialized schema formats."""
    if isinstance(feature_order, dict):
        for key in ("feature_names", "features", "feature_order", "columns"):
            if key in feature_order:
                return extract_feature_names(feature_order[key])
        raise ValueError(
            "Could not identify feature names in feature-order dictionary. "
            f"Available keys: {list(feature_order.keys())}"
        )

    if isinstance(feature_order, (list, tuple, np.ndarray, pd.Index)):
        return list(feature_order)

    raise TypeError(
        f"Unsupported feature-order format: {type(feature_order).__name__}"
    )


def load_feature_order(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature-order file not found: {path}")
    names = extract_feature_names(joblib.load(path))
    if not names:
        raise ValueError(f"No features found in {path}")
    return names


def load_required_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required model not found: {path}")
    return joblib.load(path)


print("\nLoading trained forward models...")
uts_model = load_required_model(UTS_MODEL_PATH)
density_model = load_required_model(DENSITY_MODEL_PATH)
uts_features = load_feature_order(UTS_FEATURE_ORDER_PATH)
density_features = load_feature_order(DENSITY_FEATURE_ORDER_PATH)

print("UTS model loaded.")
print("Relative-density model loaded.")
print("UTS feature schema:", uts_features)
print("Density feature schema:", density_features)


# ------------------------- FEATURE INPUTS -------------------------

def build_feature_row(feature_names, P, v, h, t):
    """
    Build one-row DataFrame in the exact feature order expected by a model.
    Add aliases here if your saved feature schemas use other column names.
    """
    aliases = {
        "p_w": P,
        "p": P,
        "power": P,
        "laser_power": P,
        "laser_power_w": P,

        "v_mm_per_s": v,
        "v": v,
        "speed": v,
        "scan_speed": v,
        "scan_speed_mm_s": v,
        "scan_speed_mm_per_s": v,

        "h_mm": h,
        "h": h,
        "hatch": h,
        "hatch_spacing": h,
        "hatch_spacing_mm": h,

        "t_mm": t,
        "t": t,
        "thickness": t,
        "layer_thickness": t,
        "layer_thickness_mm": t,
    }

    row = {}
    for feature in feature_names:
        key = str(feature).strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Cannot map model feature '{feature}' to P, v, h, or t. "
                "Update build_feature_row() with the correct mapping."
            )
        row[feature] = aliases[key]

    return pd.DataFrame([row], columns=feature_names)


def predict_scalar(model, feature_names, P, v, h, t):
    X = build_feature_row(feature_names, P, v, h, t)
    prediction = np.asarray(model.predict(X)).reshape(-1)

    if prediction.size == 0:
        raise ValueError("The model returned an empty prediction.")

    value = float(prediction[0])
    if not np.isfinite(value):
        raise ValueError(f"Model returned a non-finite prediction: {value}")
    return value


def predict_uts(P, v, h, t):
    return predict_scalar(uts_model, uts_features, P, v, h, t)


def predict_density_percent(P, v, h, t):
    raw = predict_scalar(density_model, density_features, P, v, h, t)

    # Convert fractional density (e.g. 0.995) to percentage.
    # Values already represented as percentages (e.g. 99.5) stay unchanged.
    if 0.0 <= raw <= 1.0:
        return raw * 100.0
    return raw


def calculate_energy_density(P, v, h, t):
    """Conventional volumetric energy density, J/mm^3."""
    return P / (v * h * t)


def energy_density_violation(energy_density):
    """Optuna constraints are feasible when each returned value <= 0."""
    return max(
        ENERGY_DENSITY_MIN - energy_density,
        energy_density - ENERGY_DENSITY_MAX,
        0.0,
    )


def suggest_parameters(trial):
    P = trial.suggest_float("P_W", *PARAM_BOUNDS["P_W"])
    v = trial.suggest_float("v_mm_per_s", *PARAM_BOUNDS["v_mm_per_s"])
    h = trial.suggest_float("h_mm", *PARAM_BOUNDS["h_mm"])
    return P, v, h, FIXED_LAYER_THICKNESS


# ------------------------- MODE 1: TARGET MATCHING -------------------------

def make_target_objective(target_uts, target_density):
    def objective(trial):
        P, v, h, t = suggest_parameters(trial)
        ed = calculate_energy_density(P, v, h, t)
        trial.set_user_attr("energy_density_j_mm3", ed)

        violation = energy_density_violation(ed)
        trial.set_user_attr("energy_density_violation", violation)

        # Strong penalty for out-of-window candidates; final result is
        # independently checked for feasibility below.
        if violation > 0:
            return 1e3 + violation**2

        try:
            uts = predict_uts(P, v, h, t)
            density = predict_density_percent(P, v, h, t)

            uts_rel_error = (uts - target_uts) / abs(target_uts)
            density_rel_error = (density - target_density) / abs(target_density)

            loss = (
                UTS_WEIGHT * uts_rel_error**2
                + DENSITY_WEIGHT * density_rel_error**2
            )

            trial.set_user_attr("predicted_uts_mpa", uts)
            trial.set_user_attr("predicted_density_percent", density)
            trial.set_user_attr("layer_thickness_mm", t)

            return float(loss) if np.isfinite(loss) else 1e9

        except Exception as exc:
            trial.set_user_attr("prediction_error", str(exc))
            return 1e9

    return objective


def run_target_matching():
    try:
        raw = input(f"Target UTS in MPa [{DEFAULT_TARGET_UTS_MPA}]: ").strip()
        target_uts = float(raw) if raw else DEFAULT_TARGET_UTS_MPA

        raw = input(
            f"Target relative density in % [{DEFAULT_TARGET_DENSITY_PERCENT}]: "
        ).strip()
        target_density = float(raw) if raw else DEFAULT_TARGET_DENSITY_PERCENT
    except ValueError:
        print("Invalid target. Please enter numeric values.")
        return

    if target_uts <= 0 or target_density <= 0:
        print("Targets must be greater than zero.")
        return

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print("\nRunning target-matching optimization...")
    study.optimize(
        make_target_objective(target_uts, target_density),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    feasible = [
        tr for tr in study.trials
        if tr.state == optuna.trial.TrialState.COMPLETE
        and np.isfinite(tr.value)
        and energy_density_violation(
            tr.user_attrs.get("energy_density_j_mm3", -np.inf)
        ) == 0
        and "predicted_uts_mpa" in tr.user_attrs
        and "predicted_density_percent" in tr.user_attrs
    ]

    if not feasible:
        print(
            "\nNo feasible candidate was found. Increase N_TRIALS or check "
            "parameter bounds and the energy-density window."
        )
        return

    best = min(feasible, key=lambda tr: tr.value)
    P, v, h = best.params["P_W"], best.params["v_mm_per_s"], best.params["h_mm"]
    t = FIXED_LAYER_THICKNESS
    uts = predict_uts(P, v, h, t)
    density = predict_density_percent(P, v, h, t)
    ed = calculate_energy_density(P, v, h, t)

    uts_error = uts - target_uts
    density_error = density - target_density

    print("\n" + "=" * 65)
    print("BEST FEASIBLE TARGET-MATCHING RESULT")
    print("=" * 65)
    print(f"Laser power             : {P:.6f} W")
    print(f"Scan speed              : {v:.6f} mm/s")
    print(f"Hatch spacing           : {h:.6f} mm")
    print(f"Layer thickness         : {t:.6f} mm (40 microns, fixed)")
    print(f"Volumetric energy density: {ed:.4f} J/mm^3 (constraint satisfied)")
    print(f"\nTarget UTS              : {target_uts:.4f} MPa")
    print(f"Predicted UTS           : {uts:.4f} MPa")
    print(f"UTS signed error        : {uts_error:+.4f} MPa")
    print(f"UTS absolute % error    : {abs(uts_error) / abs(target_uts) * 100:.4f}%")
    print(f"\nTarget relative density : {target_density:.4f}%")
    print(f"Predicted density       : {density:.4f}%")
    print(f"Density signed error    : {density_error:+.4f} percentage points")
    print(
        f"Density absolute % error: "
        f"{abs(density_error) / abs(target_density) * 100:.4f}%"
    )
    print(f"Objective value         : {best.value:.10f}")
    print(f"Feasible candidates     : {len(feasible)} / {N_TRIALS}")

    pd.DataFrame([{
        "mode": "target_matching",
        "P_W": P,
        "v_mm_per_s": v,
        "h_mm": h,
        "t_mm": t,
        "energy_density_J_mm3": ed,
        "target_UTS_MPa": target_uts,
        "predicted_UTS_MPa": uts,
        "UTS_signed_error_MPa": uts_error,
        "target_density_percent": target_density,
        "predicted_density_percent": density,
        "density_signed_error_percentage_points": density_error,
        "objective_value": best.value,
        "feasible_trials": len(feasible),
    }]).to_csv(OUTPUT_TARGET_MODE, index=False)

    print(f"\nSaved result to: {OUTPUT_TARGET_MODE}")


# ------------------------- MODE 2: MAXIMIZE BOTH -------------------------

def run_maximize_properties():
    # NSGA-II is a multi-objective evolutionary sampler.
    # constraints_func marks energy-window violations; feasible trials
    # have constraint values <= 0.
    def constraints_func(frozen_trial):
        ed = frozen_trial.user_attrs.get("energy_density_j_mm3", np.nan)
        if not np.isfinite(ed):
            return (1e6,)
        return (energy_density_violation(ed),)

    sampler = optuna.samplers.NSGAIISampler(
        seed=RANDOM_SEED,
        constraints_func=constraints_func,
    )
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=sampler,
    )

    def objective(trial):
        P, v, h, t = suggest_parameters(trial)
        ed = calculate_energy_density(P, v, h, t)
        trial.set_user_attr("energy_density_j_mm3", ed)
        trial.set_user_attr("layer_thickness_mm", t)

        try:
            uts = predict_uts(P, v, h, t)
            density = predict_density_percent(P, v, h, t)

            trial.set_user_attr("predicted_uts_mpa", uts)
            trial.set_user_attr("predicted_density_percent", density)

            # Do not return artificial low property values for infeasible
            # candidates; NSGA-II handles the constraint via constraints_func.
            return uts, density

        except Exception as exc:
            trial.set_user_attr("prediction_error", str(exc))
            # These values are only a fallback for failed model predictions.
            return -1e12, -1e12

    print("\nRunning multi-objective maximization...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # Keep only Pareto-front trials that satisfy the energy window and
    # have valid model predictions.
    pareto_feasible = [
        tr for tr in study.best_trials
        if tr.state == optuna.trial.TrialState.COMPLETE
        and len(tr.values) == 2
        and np.all(np.isfinite(tr.values))
        and energy_density_violation(
            tr.user_attrs.get("energy_density_j_mm3", np.nan)
        ) == 0
        and "predicted_uts_mpa" in tr.user_attrs
        and "predicted_density_percent" in tr.user_attrs
    ]

    if not pareto_feasible:
        print(
            "\nNo feasible Pareto candidates were found. Increase N_TRIALS "
            "or check the bounds and energy-density window."
        )
        return

    rows = []
    for tr in pareto_feasible:
        P = tr.params["P_W"]
        v = tr.params["v_mm_per_s"]
        h = tr.params["h_mm"]
        t = FIXED_LAYER_THICKNESS
        ed = calculate_energy_density(P, v, h, t)
        rows.append({
            "P_W": P,
            "v_mm_per_s": v,
            "h_mm": h,
            "t_mm": t,
            "energy_density_J_mm3": ed,
            "predicted_UTS_MPa": tr.user_attrs["predicted_uts_mpa"],
            "predicted_density_percent": tr.user_attrs["predicted_density_percent"],
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        ["predicted_UTS_MPa", "predicted_density_percent"],
        ascending=[False, False],
    ).reset_index(drop=True)
    result_df.to_csv(OUTPUT_MAX_MODE, index=False)

    print("\n" + "=" * 65)
    print("FEASIBLE PARETO-OPTIMAL CANDIDATES")
    print("=" * 65)
    print(
        "These candidates represent trade-offs; no single candidate is "
        "necessarily best for both properties."
    )
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved {len(result_df)} Pareto candidates to: {OUTPUT_MAX_MODE}")


# ------------------------- MAIN -------------------------

def main():
    print("\n" + "=" * 65)
    print("LPBF INVERSE OPTIMIZER")
    print("=" * 65)
    print(f"Fixed layer thickness : {FIXED_LAYER_THICKNESS:.4f} mm (40 microns)")
    print(
        f"Energy-density window : {ENERGY_DENSITY_MIN:.1f}–"
        f"{ENERGY_DENSITY_MAX:.1f} J/mm^3"
    )
    print(f"Trials per run        : {N_TRIALS}")
    print("\nChoose optimization mode:")
    print("1. Match target UTS and relative density")
    print("2. Maximize UTS and relative density (Pareto front)")

    choice = input("Enter 1 or 2 [1]: ").strip() or "1"

    if choice == "1":
        run_target_matching()
    elif choice == "2":
        run_maximize_properties()
    else:
        print("Invalid choice. Run again and enter 1 or 2.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        sys.exit(130)
