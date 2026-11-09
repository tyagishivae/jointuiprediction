import joblib
import optuna
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"

# ------------------------------------------------------------
# TARGET PROPERTIES
# Change these values for each inverse optimization
# ------------------------------------------------------------

TARGET_UTS = 900.0             # MPa
TARGET_DENSITY = 99.5          # %

# Equal importance to both target properties
UTS_WEIGHT = 0.50
DENSITY_WEIGHT = 0.50

# Number of Optuna trials
N_TRIALS = 300


# ============================================================
# PROCESS PARAMETER BOUNDS
# ============================================================

PARAM_BOUNDS = {

    "P_W": (150.0, 370.0),

    "v_mm_per_s": (350.0, 1772.0),

    "h_mm": (0.09, 0.12),

    "t_mm": (0.03, 0.05),
}


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")
print("-" * 70)

UTS_MODEL = joblib.load(
    MODEL_DIR / "UTS_MPa.pkl"
)

DENSITY_MODEL = joblib.load(
    MODEL_DIR / "RelativeDensity_frac.pkl"
)

# General feature order
FEATURE_ORDER = joblib.load(
    MODEL_DIR / "feature_order.pkl"
)

# UTS-specific feature order
UTS_FEATURE_ORDER = joblib.load(
    MODEL_DIR / "UTS_feature_order.pkl"
)

print("UTS model                : Loaded")
print("Relative density model  : Loaded")
print("General feature order    : Loaded")
print("UTS feature order        : Loaded")


# ============================================================
# OPTIONAL MODELS
# ============================================================

MELT_WIDTH_MODEL = None
MELT_DEPTH_MODEL = None
DEFECT_MODEL = None

try:
    MELT_WIDTH_MODEL = joblib.load(
        MODEL_DIR / "MeltPoolWidth_um.pkl"
    )
    print("Melt pool width model    : Loaded")
except Exception:
    print("Melt pool width model    : Not loaded")

try:
    MELT_DEPTH_MODEL = joblib.load(
        MODEL_DIR / "MeltPoolDepth_um.pkl"
    )
    print("Melt pool depth model    : Loaded")
except Exception:
    print("Melt pool depth model    : Not loaded")

try:
    DEFECT_MODEL = joblib.load(
        MODEL_DIR / "defect_classifier.pkl"
    )
    print("Defect classifier        : Loaded")
except Exception:
    print("Defect classifier        : Not loaded")


# ============================================================
# FEATURE ORDER HELPER
# ============================================================

def extract_feature_names(order_object):
    """
    Convert different possible feature-order formats
    into a simple list of feature names.
    """

    if isinstance(order_object, (list, tuple, np.ndarray)):
        return list(order_object)

    if isinstance(order_object, dict):

        # Common possible formats
        for key in [
            "features",
            "feature_order",
            "columns",
            "feature_names"
        ]:

            if key in order_object:
                value = order_object[key]

                if isinstance(
                    value,
                    (list, tuple, np.ndarray)
                ):
                    return list(value)

        # If dictionary itself represents ordering
        return list(order_object.keys())

    raise TypeError(
        f"Unsupported feature-order format: "
        f"{type(order_object)}"
    )


GENERAL_FEATURES = extract_feature_names(
    FEATURE_ORDER
)

UTS_FEATURES = extract_feature_names(
    UTS_FEATURE_ORDER
)


print("\nGeneral feature order:")
print(GENERAL_FEATURES)

print("\nUTS feature order:")
print(UTS_FEATURES)


# ============================================================
# PROCESS PARAMETER DICTIONARY
# ============================================================

def parameter_dictionary(P, v, h, t):

    return {

        "P_W": P,

        "v_mm_per_s": v,

        "h_mm": h,

        "t_mm": t,
    }


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_input(
    parameters,
    feature_names
):
    """
    Create a DataFrame in exactly the feature order
    required by a model.
    """

    missing = [
        f
        for f in feature_names
        if f not in parameters
    ]

    if missing:

        raise ValueError(
            "The model requires feature(s) that are "
            f"not available: {missing}\n"
            f"Available parameters: {list(parameters.keys())}"
        )

    return pd.DataFrame(
        [[parameters[f] for f in feature_names]],
        columns=feature_names
    )


# ============================================================
# RELATIVE DENSITY CONVERSION
# ============================================================

def density_to_percent(value):
    """
    RelativeDensity_frac.pkl is expected to return a fraction,
    e.g. 0.995.

    This function converts it to percentage:
    0.995 -> 99.5

    If a model already returns percentage:
    99.5 -> 99.5
    """

    value = float(value)

    if value <= 1.5:
        return value * 100.0

    return value


# ============================================================
# FORWARD PREDICTION
# ============================================================

def predict_properties(
    P,
    v,
    h,
    t
):

    parameters = parameter_dictionary(
        P,
        v,
        h,
        t
    )

    # --------------------------------------------------------
    # UTS
    # --------------------------------------------------------

    X_uts = create_input(
        parameters,
        UTS_FEATURES
    )

    uts_prediction = float(
        UTS_MODEL.predict(X_uts)[0]
    )

    # --------------------------------------------------------
    # Relative Density
    # --------------------------------------------------------

    X_density = create_input(
        parameters,
        GENERAL_FEATURES
    )

    density_raw = float(
        DENSITY_MODEL.predict(X_density)[0]
    )

    density_prediction = density_to_percent(
        density_raw
    )

    return {
        "UTS": uts_prediction,
        "Density": density_prediction
    }


# ============================================================
# OPTIONAL PROCESS PREDICTIONS
# ============================================================

def predict_process_outputs(
    P,
    v,
    h,
    t
):

    parameters = parameter_dictionary(
        P,
        v,
        h,
        t
    )

    outputs = {}

    # --------------------------------------------------------
    # Melt Pool Width
    # --------------------------------------------------------

    if MELT_WIDTH_MODEL is not None:

        try:

            X = create_input(
                parameters,
                GENERAL_FEATURES
            )

            outputs["MeltPoolWidth"] = float(
                MELT_WIDTH_MODEL.predict(X)[0]
            )

        except Exception as e:

            outputs["MeltPoolWidth"] = (
                f"Prediction failed: {e}"
            )

    # --------------------------------------------------------
    # Melt Pool Depth
    # --------------------------------------------------------

    if MELT_DEPTH_MODEL is not None:

        try:

            X = create_input(
                parameters,
                GENERAL_FEATURES
            )

            outputs["MeltPoolDepth"] = float(
                MELT_DEPTH_MODEL.predict(X)[0]
            )

        except Exception as e:

            outputs["MeltPoolDepth"] = (
                f"Prediction failed: {e}"
            )

    # --------------------------------------------------------
    # Defect Class
    # --------------------------------------------------------

    if DEFECT_MODEL is not None:

        try:

            X = create_input(
                parameters,
                GENERAL_FEATURES
            )

            outputs["DefectClass"] = (
                DEFECT_MODEL.predict(X)[0]
            )

        except Exception as e:

            outputs["DefectClass"] = (
                f"Prediction failed: {e}"
            )

    return outputs


# ============================================================
# PRE-FLIGHT MODEL TEST
# ============================================================

print("\nTesting forward models...")
print("-" * 70)

test_P = (
    PARAM_BOUNDS["P_W"][0]
    + PARAM_BOUNDS["P_W"][1]
) / 2

test_v = (
    PARAM_BOUNDS["v_mm_per_s"][0]
    + PARAM_BOUNDS["v_mm_per_s"][1]
) / 2

test_h = (
    PARAM_BOUNDS["h_mm"][0]
    + PARAM_BOUNDS["h_mm"][1]
) / 2

test_t = (
    PARAM_BOUNDS["t_mm"][0]
    + PARAM_BOUNDS["t_mm"][1]
) / 2


try:

    test_prediction = predict_properties(
        test_P,
        test_v,
        test_h,
        test_t
    )

    print(
        f"Test UTS      : "
        f"{test_prediction['UTS']:.3f} MPa"
    )

    print(
        f"Test Density  : "
        f"{test_prediction['Density']:.3f} %"
    )

    print("Forward model test: PASS")

except Exception as e:

    print("\nForward model test FAILED.")
    print("\nThis usually means the feature order or")
    print("model input schema does not match.")
    print("\nError:")
    print(e)

    raise SystemExit(
        "\nOptimization stopped."
    )


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # Suggest process parameters
    # --------------------------------------------------------

    P = trial.suggest_float(
        "P_W",
        *PARAM_BOUNDS["P_W"]
    )

    v = trial.suggest_float(
        "v_mm_per_s",
        *PARAM_BOUNDS["v_mm_per_s"]
    )

    h = trial.suggest_float(
        "h_mm",
        *PARAM_BOUNDS["h_mm"]
    )

    t = trial.suggest_float(
        "t_mm",
        *PARAM_BOUNDS["t_mm"]
    )

    # --------------------------------------------------------
    # Forward prediction
    # --------------------------------------------------------

    try:

        prediction = predict_properties(
            P,
            v,
            h,
            t
        )

    except Exception:

        return float("inf")

    # --------------------------------------------------------
    # Property errors
    # --------------------------------------------------------

    uts_error = (
        prediction["UTS"]
        - TARGET_UTS
    )

    density_error = (
        prediction["Density"]
        - TARGET_DENSITY
    )

    # --------------------------------------------------------
    # Relative errors
    #
    # This prevents UTS (MPa) from dominating density (%)
    # simply because of numerical scale.
    # --------------------------------------------------------

    uts_relative_error = (
        uts_error / TARGET_UTS
    )

    density_relative_error = (
        density_error / TARGET_DENSITY
    )

    # --------------------------------------------------------
    # Combined objective
    # --------------------------------------------------------

    score = (

        UTS_WEIGHT
        * uts_relative_error ** 2

        +

        DENSITY_WEIGHT
        * density_relative_error ** 2
    )

    return score


# ============================================================
# RUN OPTIMIZATION
# ============================================================

print("\n")
print("=" * 70)
print("STARTING INVERSE OPTIMIZATION")
print("=" * 70)

print(
    f"\nTarget UTS     : "
    f"{TARGET_UTS:.3f} MPa"
)

print(
    f"Target Density : "
    f"{TARGET_DENSITY:.3f} %"
)

print(
    f"Trials         : "
    f"{N_TRIALS}"
)

print()


study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)


# ============================================================
# BEST PARAMETERS
# ============================================================

best_parameters = study.best_params

BEST_P = best_parameters["P_W"]
BEST_V = best_parameters["v_mm_per_s"]
BEST_H = best_parameters["h_mm"]
BEST_T = best_parameters["t_mm"]


# ============================================================
# FINAL FORWARD PREDICTION
# ============================================================

best_prediction = predict_properties(
    BEST_P,
    BEST_V,
    BEST_H,
    BEST_T
)


# ============================================================
# CALCULATE ERRORS
# ============================================================

final_uts = best_prediction["UTS"]
final_density = best_prediction["Density"]


uts_error = (
    final_uts
    - TARGET_UTS
)

density_error = (
    final_density
    - TARGET_DENSITY
)


uts_absolute_error = abs(
    uts_error
)

density_absolute_error = abs(
    density_error
)


uts_percentage_error = (
    uts_absolute_error
    / TARGET_UTS
) * 100


density_percentage_error = (
    density_absolute_error
    / TARGET_DENSITY
) * 100


# ============================================================
# OPTIONAL PROCESS OUTPUTS
# ============================================================

process_outputs = predict_process_outputs(
    BEST_P,
    BEST_V,
    BEST_H,
    BEST_T
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("INVERSE OPTIMIZATION RESULT")
print("=" * 70)


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

print("\nTARGET PROPERTIES")
print("-" * 70)

print(
    f"Target UTS                 : "
    f"{TARGET_UTS:.4f} MPa"
)

print(
    f"Target Relative Density   : "
    f"{TARGET_DENSITY:.4f} %"
)


# ------------------------------------------------------------
# OPTIMIZED PARAMETERS
# ------------------------------------------------------------

print("\nOPTIMIZED PROCESS PARAMETERS")
print("-" * 70)

print(
    f"Laser Power               : "
    f"{BEST_P:.6f} W"
)

print(
    f"Scan Speed                : "
    f"{BEST_V:.6f} mm/s"
)

print(
    f"Hatch Spacing             : "
    f"{BEST_H:.6f} mm"
)

print(
    f"Layer Thickness           : "
    f"{BEST_T:.6f} mm"
)


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print("\nPREDICTED PROPERTIES")
print("-" * 70)

print(
    f"Predicted UTS             : "
    f"{final_uts:.4f} MPa"
)

print(
    f"Predicted Relative Density: "
    f"{final_density:.4f} %"
)


# ------------------------------------------------------------
# ERRORS
# ------------------------------------------------------------

print("\nPREDICTION ERRORS")
print("-" * 70)

print(
    f"UTS signed error          : "
    f"{uts_error:+.4f} MPa"
)

print(
    f"UTS absolute error        : "
    f"{uts_absolute_error:.4f} MPa"
)

print(
    f"UTS percentage error      : "
    f"{uts_percentage_error:.4f} %"
)

print()

print(
    f"Density signed error      : "
    f"{density_error:+.4f} percentage points"
)

print(
    f"Density absolute error    : "
    f"{density_absolute_error:.4f} percentage points"
)

print(
    f"Density percentage error  : "
    f"{density_percentage_error:.4f} %"
)


# ------------------------------------------------------------
# OPTIONAL OUTPUTS
# ------------------------------------------------------------

if process_outputs:

    print("\nADDITIONAL PROCESS PREDICTIONS")
    print("-" * 70)

    for name, value in process_outputs.items():

        print(
            f"{name:<27}: {value}"
        )


# ------------------------------------------------------------
# OPTIMIZATION INFO
# ------------------------------------------------------------

print("\nOPTIMIZATION INFORMATION")
print("-" * 70)

print(
    f"Number of trials          : "
    f"{len(study.trials)}"
)

print(
    f"Best objective            : "
    f"{study.best_value:.10f}"
)


print("\n")
print("=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)