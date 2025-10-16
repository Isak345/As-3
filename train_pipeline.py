"""
Training pipeline for diabetes progression prediction.
Uses scikit-learn diabetes dataset as a stand-in for EHR data.
"""

import pickle
import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(test_size=0.2, random_state=42):
    """Load and split diabetes dataset."""
    print("Loading diabetes dataset...")
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type="linear", **model_params):
    """Train model with standardization."""
    print(f"Training {model_type} model with params: {model_params}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Select model
    if model_type == "linear":
        model = LinearRegression(**model_params)
    elif model_type == "ridge":
        model = Ridge(**model_params)
    elif model_type == "random_forest":
        model = RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    return scaler, model


def evaluate_model(scaler, model, X_test, y_test):
    """Evaluate model and return metrics."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }
    
    print(f"\nModel Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    return metrics, y_pred


def calculate_risk_calibration(y_test, y_pred, threshold_percentile=75):
    """
    Calculate precision/recall for high-risk classification.
    Uses threshold at given percentile of actual progression values.
    """
    threshold = np.percentile(y_test, threshold_percentile)
    
    y_true_high_risk = (y_test >= threshold).astype(int)
    y_pred_high_risk = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    tp = np.sum((y_true_high_risk == 1) & (y_pred_high_risk == 1))
    fp = np.sum((y_true_high_risk == 0) & (y_pred_high_risk == 1))
    fn = np.sum((y_true_high_risk == 1) & (y_pred_high_risk == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    calibration = {
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    
    print(f"\nRisk Calibration (threshold={threshold:.2f}, p{threshold_percentile}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return calibration


def save_artifacts(scaler, model, metrics, version, output_dir="models"):
    """Save model artifacts and metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save model
    model_file = output_path / f"model_{version}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump({"scaler": scaler, "model": model}, f)
    print(f"\nModel saved to: {model_file}")
    
    # Save metrics
    metrics_file = output_path / f"metrics_{version}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    return str(model_file), str(metrics_file)


def main():
    parser = argparse.ArgumentParser(description="Train diabetes progression model")
    parser.add_argument("--version", type=str, required=True, help="Model version (e.g., v0.1)")
    parser.add_argument("--model-type", type=str, default="linear", 
                       choices=["linear", "ridge", "random_forest"],
                       help="Model type to train")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    
    # Model-specific parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha parameter")
    parser.add_argument("--n-estimators", type=int, default=100, help="RF n_estimators")
    parser.add_argument("--max-depth", type=int, default=None, help="RF max_depth")
    
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Prepare model parameters
    model_params = {}
    if args.model_type == "ridge":
        model_params["alpha"] = args.alpha
        model_params["random_state"] = args.random_state
    elif args.model_type == "random_forest":
        model_params["n_estimators"] = args.n_estimators
        model_params["max_depth"] = args.max_depth
        model_params["random_state"] = args.random_state
    
    # Train model
    scaler, model = train_model(X_train, y_train, args.model_type, **model_params)
    
    # Evaluate
    metrics, y_pred = evaluate_model(scaler, model, X_test, y_test)
    
    # Calculate risk calibration
    calibration = calculate_risk_calibration(y_test, y_pred)
    metrics["calibration"] = calibration
    
    # Add metadata
    metrics["version"] = args.version
    metrics["model_type"] = args.model_type
    metrics["model_params"] = model_params
    metrics["test_size"] = args.test_size
    
    # Save artifacts
    save_artifacts(scaler, model, metrics, args.version, args.output_dir)
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
