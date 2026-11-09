"""Joint inverse optimization for hardness, UTS, and relative density."""

from pathlib import Path
import argparse

import joblib
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FEATURES_PATH = MODEL_DIR / "joint_500_feature_order.pkl"
MODEL_PATHS = {
    "Hardness_HV": MODEL_DIR / "Hardness_HV_joint_500.pkl",
    "UTS_MPa": MODEL_DIR / "UTS_MPa_joint_500.pkl",
    "RelativeDensity_pct": MODEL_DIR / "RelativeDensity_pct_joint_500.pkl",
}
TARGETS = list(MODEL_PATHS)
HARDNESS_DATA_PATH = BASE_DIR / "hardness_experimental.csv"
UTS_DENSITY_DATA_PATH = BASE_DIR / "uts_density_experimental.csv"
VALIDATION_RESULTS_PATH = BASE_DIR / "joint_holdout_validation_results.csv"
VALIDATION_TARGETS_PATH = BASE_DIR / "joint_validation_conditions.csv"
VALIDATION_MODEL_DIR = MODEL_DIR / "joint_validation_fold"
VALIDATION_MODEL_PATHS = {
    target: VALIDATION_MODEL_DIR / f"{target}_joint_validation.pkl"
    for target in TARGETS
}
VALIDATION_FEATURES_PATH = (
    VALIDATION_MODEL_DIR / "joint_validation_feature_order.pkl"
)
OUTPUT_PATH = BASE_DIR / "joint_inverse_pareto_candidates.csv"
TARGET_MATCH_OUTPUT_PATH = BASE_DIR / "joint_inverse_target_match.csv"
HOLDOUT_INVERSE_OUTPUT_PATH = BASE_DIR / "joint_inverse_holdout_recovery.csv"

N_TRIALS = 1500
N_TARGET_TRIALS = 600
RANDOM_SEED = 42
N_CANDIDATES = 6


for required_path in [
    FEATURES_PATH,
    *MODEL_PATHS.values(),
    HARDNESS_DATA_PATH,
    UTS_DENSITY_DATA_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input not found: {required_path}")

FEATURES = joblib.load(FEATURES_PATH)
MODELS = {
    target: joblib.load(path)
    for target, path in MODEL_PATHS.items()
}

hardness_data = pd.read_csv(HARDNESS_DATA_PATH)
uts_density_data = pd.read_csv(UTS_DENSITY_DATA_PATH)
process_features = [
    "LaserPower_W",
    "ScanSpeed_mm_s",
    "HatchDistance_mm",
    "LayerThickness_mm",
]
experimental = hardness_data.merge(
    uts_density_data,
    on="Sample",
    how="inner",
    validate="one_to_one",
)
conditions = (
    experimental
    .groupby(process_features, as_index=False)
    .agg(
        **{target: (target, "mean") for target in TARGETS},
        Split=("Split", "first"),
    )
)
training_conditions = conditions.loc[
    conditions["Split"] == "train"
].reset_index(drop=True)

PARAM_BOUNDS = {
    "LaserPower_W": (
        float(conditions["LaserPower_W"].min()),
        float(conditions["LaserPower_W"].max()),
    ),
    "ScanSpeed_mm_s": (
        float(conditions["ScanSpeed_mm_s"].min()),
        float(conditions["ScanSpeed_mm_s"].max()),
    ),
    "HatchDistance_mm": (
        float(conditions["HatchDistance_mm"].min()),
        float(conditions["HatchDistance_mm"].max()),
    ),
}
FIXED_LAYER_THICKNESS = float(conditions["LayerThickness_mm"].iloc[0])

condition_ved = (
    conditions["LaserPower_W"]
    / (
        conditions["ScanSpeed_mm_s"]
        * conditions["HatchDistance_mm"]
        * conditions["LayerThickness_mm"]
    )
)
ENERGY_DENSITY_BOUNDS = (
    float(condition_ved.min()),
    float(condition_ved.max()),
)

TRAIN_PARAM_BOUNDS = {
    feature: (
        float(training_conditions[feature].min()),
        float(training_conditions[feature].max()),
    )
    for feature in PARAM_BOUNDS
}
training_ved = (
    training_conditions["LaserPower_W"]
    / (
        training_conditions["ScanSpeed_mm_s"]
        * training_conditions["HatchDistance_mm"]
        * training_conditions["LayerThickness_mm"]
    )
)
TRAIN_ENERGY_DENSITY_BOUNDS = (
    float(training_ved.min()),
    float(training_ved.max()),
)


if not all(hasattr(model, "predict") for model in MODELS.values()):
    raise TypeError("Every loaded joint model must provide predict().")
if len(FEATURES) != 4:
    raise ValueError(f"Expected four model input features, found {FEATURES!r}.")


def predict_properties(parameters, models=None):
    models = MODELS if models is None else models
    model_input = pd.DataFrame(
        [[parameters[feature] for feature in FEATURES]],
        columns=FEATURES,
    )
    predictions = {
        target: float(model.predict(model_input)[0])
        for target, model in models.items()
    }
    if not all(np.isfinite(value) for value in predictions.values()):
        raise ValueError(f"Non-finite model prediction: {predictions}")
    return predictions


def signed_percent_error(predicted, actual):
    if abs(actual) <= np.finfo(float).eps:
        return np.nan
    return 100.0 * (predicted - actual) / abs(actual)


def show_real_holdout_check():
    if not VALIDATION_RESULTS_PATH.exists():
        print(f"Validation report not found: {VALIDATION_RESULTS_PATH}")
        return

    validation = pd.read_csv(VALIDATION_RESULTS_PATH)
    print("\nPreviously completed real-data holdout check")
    print("These are forward-model checks at two withheld settings.")
    print(validation.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def optimize_for_targets(
    target_values,
    models,
    parameter_bounds,
    energy_density_bounds,
    target_scales,
    seed,
    n_trials,
):
    def objective(trial):
        parameters = {
            name: trial.suggest_float(name, *bounds)
            for name, bounds in parameter_bounds.items()
        }
        parameters["LayerThickness_mm"] = FIXED_LAYER_THICKNESS
        energy_density = parameters["LaserPower_W"] / (
            parameters["ScanSpeed_mm_s"]
            * parameters["HatchDistance_mm"]
            * FIXED_LAYER_THICKNESS
        )
        trial.set_user_attr("energy_density_J_mm3", energy_density)
        lower, upper = energy_density_bounds
        violation = max(lower - energy_density, energy_density - upper, 0.0)
        if violation > 0:
            return 1e3 + violation**2

        predictions = predict_properties(parameters, models)
        loss = np.mean([
            ((predictions[target] - target_values[target]) / target_scales[target]) ** 2
            for target in TARGETS
        ])
        trial.set_user_attr("predictions", predictions)
        return float(loss) if np.isfinite(loss) else 1e9

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    parameters = {
        **best.params,
        "LayerThickness_mm": FIXED_LAYER_THICKNESS,
    }
    predictions = predict_properties(parameters, models)
    return parameters, predictions, float(best.value)


def read_target_values_from_prompt():
    target_values = {}
    for target in TARGETS:
        units = "HV" if target == "Hardness_HV" else (
            "MPa" if target == "UTS_MPa" else "%"
        )
        raw = input(f"Target {target} ({units}): ").strip()
        try:
            target_values[target] = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric target for {target}: {raw!r}") from exc
    return target_values


def run_target_matching(target_values=None):
    if target_values is None:
        target_values = read_target_values_from_prompt()
    target_scales = (
        conditions[TARGETS].max() - conditions[TARGETS].min()
    ).replace(0, 1.0).to_dict()
    parameters, predictions, loss = optimize_for_targets(
        target_values=target_values,
        models=MODELS,
        parameter_bounds=PARAM_BOUNDS,
        energy_density_bounds=ENERGY_DENSITY_BOUNDS,
        target_scales=target_scales,
        seed=RANDOM_SEED,
        n_trials=N_TARGET_TRIALS,
    )
    energy_density = parameters["LaserPower_W"] / (
        parameters["ScanSpeed_mm_s"]
        * parameters["HatchDistance_mm"]
        * parameters["LayerThickness_mm"]
    )
    result = {
        "LaserPower_W": parameters["LaserPower_W"],
        "ScanSpeed_mm_s": parameters["ScanSpeed_mm_s"],
        "HatchDistance_mm": parameters["HatchDistance_mm"],
        "LayerThickness_mm": parameters["LayerThickness_mm"],
        "VED_J_mm3": energy_density,
        "NormalizedTargetLoss": loss,
    }
    for target in TARGETS:
        result[f"Target_{target}"] = target_values[target]
        result[f"Predicted_{target}"] = predictions[target]
        result[f"Error_{target}"] = predictions[target] - target_values[target]
        result[f"ErrorPercent_{target}"] = signed_percent_error(
            predictions[target],
            target_values[target],
        )
        result[f"AbsoluteErrorPercent_{target}"] = abs(
            result[f"ErrorPercent_{target}"]
        )
    result_frame = pd.DataFrame([result])
    result_frame.to_csv(TARGET_MATCH_OUTPUT_PATH, index=False)
    print("\nBest target-matching parameter set:")
    print(result_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved target-matching result to: {TARGET_MATCH_OUTPUT_PATH}")


def run_holdout_inverse_recovery():
    required_paths = [
        VALIDATION_TARGETS_PATH,
        VALIDATION_FEATURES_PATH,
        *VALIDATION_MODEL_PATHS.values(),
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run the grouped validation section in training.ipynb first; "
            f"missing validation artifact(s): {missing}"
        )

    validation_features = joblib.load(VALIDATION_FEATURES_PATH)
    if list(validation_features) != list(FEATURES):
        raise ValueError("Validation and final model feature orders do not match.")
    validation_models = {
        target: joblib.load(path)
        for target, path in VALIDATION_MODEL_PATHS.items()
    }
    validation_conditions = pd.read_csv(VALIDATION_TARGETS_PATH)
    target_scales = (
        training_conditions[TARGETS].max()
        - training_conditions[TARGETS].min()
    ).replace(0, 1.0).to_dict()

    results = []
    for holdout_index, condition in validation_conditions.iterrows():
        target_values = {
            target: float(condition[target])
            for target in TARGETS
        }
        parameters, predictions, loss = optimize_for_targets(
            target_values=target_values,
            models=validation_models,
            parameter_bounds=TRAIN_PARAM_BOUNDS,
            energy_density_bounds=TRAIN_ENERGY_DENSITY_BOUNDS,
            target_scales=target_scales,
            seed=RANDOM_SEED + holdout_index,
            n_trials=N_TARGET_TRIALS,
        )
        result = {
            "ValidationCondition": holdout_index + 1,
            "Actual_LaserPower_W": condition["LaserPower_W"],
            "Actual_ScanSpeed_mm_s": condition["ScanSpeed_mm_s"],
            "Actual_HatchDistance_mm": condition["HatchDistance_mm"],
            "Actual_LayerThickness_mm": condition["LayerThickness_mm"],
            "Inverse_LaserPower_W": parameters["LaserPower_W"],
            "Inverse_ScanSpeed_mm_s": parameters["ScanSpeed_mm_s"],
            "Inverse_HatchDistance_mm": parameters["HatchDistance_mm"],
            "Inverse_LayerThickness_mm": parameters["LayerThickness_mm"],
            "NormalizedTargetLoss": loss,
        }
        for feature in process_features:
            parameter_error = parameters[feature] - condition[feature]
            result[f"AbsoluteParameterError_{feature}"] = abs(
                parameter_error
            )
            result[f"ParameterErrorPercent_{feature}"] = signed_percent_error(
                parameters[feature],
                condition[feature],
            )
            result[f"AbsoluteParameterErrorPercent_{feature}"] = abs(
                result[f"ParameterErrorPercent_{feature}"]
            )
        for target in TARGETS:
            result[f"Measured_{target}"] = target_values[target]
            result[f"InversePredicted_{target}"] = predictions[target]
            result[f"PropertyError_{target}"] = predictions[target] - target_values[target]
            result[f"PropertyErrorPercent_{target}"] = signed_percent_error(
                predictions[target],
                target_values[target],
            )
            result[f"AbsolutePropertyErrorPercent_{target}"] = abs(
                result[f"PropertyErrorPercent_{target}"]
            )
        results.append(result)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(HOLDOUT_INVERSE_OUTPUT_PATH, index=False)
    print("\nInverse recovery against the two held-out conditions:")
    print(result_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nThis compares recovered settings with known settings using models "
        "trained only on the other 13 conditions."
    )
    print(f"Saved inverse-recovery results to: {HOLDOUT_INVERSE_OUTPUT_PATH}")


def generate_candidates():
    def energy_density_violation(trial):
        power = trial.params["LaserPower_W"]
        speed = trial.params["ScanSpeed_mm_s"]
        hatch = trial.params["HatchDistance_mm"]
        energy_density = power / (
            speed * hatch * FIXED_LAYER_THICKNESS
        )
        trial.set_user_attr("energy_density_J_mm3", energy_density)
        lower, upper = ENERGY_DENSITY_BOUNDS
        return (max(lower - energy_density, energy_density - upper, 0.0),)

    sampler = optuna.samplers.NSGAIISampler(
        seed=RANDOM_SEED,
        constraints_func=energy_density_violation,
    )
    study = optuna.create_study(
        directions=["maximize"] * len(TARGETS),
        sampler=sampler,
    )

    def objective(trial):
        parameters = {
            name: trial.suggest_float(name, *bounds)
            for name, bounds in PARAM_BOUNDS.items()
        }
        parameters["LayerThickness_mm"] = FIXED_LAYER_THICKNESS
        energy_density = parameters["LaserPower_W"] / (
            parameters["ScanSpeed_mm_s"]
            * parameters["HatchDistance_mm"]
            * FIXED_LAYER_THICKNESS
        )
        trial.set_user_attr("energy_density_J_mm3", energy_density)

        predictions = predict_properties(parameters)
        for target, value in predictions.items():
            trial.set_user_attr(target, value)
        return tuple(predictions[target] for target in TARGETS)

    print(
        f"\nSearching {N_TRIALS} trials in measured bounds: "
        f"P={PARAM_BOUNDS['LaserPower_W']}, "
        f"v={PARAM_BOUNDS['ScanSpeed_mm_s']}, "
        f"h={PARAM_BOUNDS['HatchDistance_mm']}, "
        f"t={FIXED_LAYER_THICKNESS} mm, "
        f"VED={ENERGY_DENSITY_BOUNDS} J/mm^3"
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    rows = []
    for trial in study.best_trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        parameters = {
            **trial.params,
            "LayerThickness_mm": FIXED_LAYER_THICKNESS,
        }
        energy_density = trial.user_attrs.get("energy_density_J_mm3", np.nan)
        lower, upper = ENERGY_DENSITY_BOUNDS
        if not np.isfinite(energy_density) or not lower <= energy_density <= upper:
            continue
        predictions = predict_properties(parameters)
        rows.append({
            **parameters,
            "VED_J_mm3": energy_density,
            **predictions,
        })

    pareto = pd.DataFrame(rows).drop_duplicates(process_features)
    if pareto.empty:
        raise RuntimeError("The search produced no feasible Pareto candidates.")

    value_matrix = pareto[TARGETS].to_numpy(dtype=float)
    non_dominated = np.ones(len(pareto), dtype=bool)
    for index, value in enumerate(value_matrix):
        dominated = np.all(value_matrix >= value, axis=1) & np.any(
            value_matrix > value,
            axis=1,
        )
        if dominated.any():
            non_dominated[index] = False
    pareto = pareto.loc[non_dominated].reset_index(drop=True)

    observed_min = conditions[TARGETS].min()
    observed_span = (conditions[TARGETS].max() - observed_min).replace(0, 1.0)
    normalized_properties = (pareto[TARGETS] - observed_min) / observed_span
    pareto["BalancedScore"] = normalized_properties.mean(axis=1)
    pareto = pareto.sort_values("BalancedScore", ascending=False).reset_index(drop=True)

    candidate_count = min(N_CANDIDATES, len(pareto))
    parameter_span = (
        pareto[process_features].max() - pareto[process_features].min()
    ).replace(0, 1.0)
    normalized_parameters = (pareto[process_features] - pareto[process_features].min()) / parameter_span

    selected = [0]
    while len(selected) < candidate_count:
        remaining = [index for index in range(len(pareto)) if index not in selected]
        distance_matrix = np.linalg.norm(
            normalized_parameters.iloc[remaining].to_numpy()[:, None, :]
            - normalized_parameters.iloc[selected].to_numpy()[None, :, :],
            axis=2,
        )
        selected.append(remaining[int(distance_matrix.min(axis=1).argmax())])

    candidates = pareto.iloc[selected].sort_values(
        "BalancedScore",
        ascending=False,
    ).reset_index(drop=True)
    candidates.insert(0, "Candidate", np.arange(1, len(candidates) + 1))
    candidates["Interpretation"] = (
        "Surrogate prediction; confirm by manufacturing and measurement"
    )
    candidates.to_csv(OUTPUT_PATH, index=False)

    print("\nSix diverse non-dominated candidates:")
    print(candidates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved candidates to: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Joint inverse optimizer for hardness, UTS, and relative density."
    )
    parser.add_argument(
        "--mode",
        choices=("targets", "maximize", "validate-inverse"),
        help="Run target matching, Pareto maximization, or holdout inverse recovery.",
    )
    parser.add_argument("--hardness-target", type=float)
    parser.add_argument("--uts-target", type=float)
    parser.add_argument("--density-target", type=float)
    args = parser.parse_args()

    print("Joint inverse optimizer: hardness + UTS + relative density")
    print(f"Loaded properties: {', '.join(TARGETS)}")
    print(f"Fixed layer thickness: {FIXED_LAYER_THICKNESS:.4f} mm")
    show_real_holdout_check()

    mode = args.mode
    if mode is None:
        print("\nChoose a mode:")
        print("1. Find parameters for requested property targets")
        print("2. Maximize all three properties (Pareto candidates)")
        print("3. Test inverse recovery on the two held-out conditions")
        choice = input("Enter 1, 2, or 3 [2]: ").strip() or "2"
        mode = {
            "1": "targets",
            "2": "maximize",
            "3": "validate-inverse",
        }.get(choice)

    if mode == "targets":
        supplied_targets = [
            args.hardness_target,
            args.uts_target,
            args.density_target,
        ]
        if any(value is not None for value in supplied_targets):
            if not all(value is not None for value in supplied_targets):
                parser.error(
                    "provide all three target options together: "
                    "--hardness-target, --uts-target, --density-target"
                )
            run_target_matching(dict(zip(TARGETS, supplied_targets)))
        else:
            run_target_matching()
    elif mode == "maximize":
        generate_candidates()
    elif mode == "validate-inverse":
        run_holdout_inverse_recovery()
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
