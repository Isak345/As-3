#!/bin/bash
# Script to train and compare models for Iteration 2 (v0.2)

set -e

echo "🚀 Starting Iteration 2 - Model Comparison"
echo "=========================================="

# Create models directory
mkdir -p models

# Baseline (v0.1) for comparison
echo ""
echo "📊 Training Baseline (v0.1 - Linear Regression)..."
python train.py \
  --version v0.1 \
  --model-type linear \
  --output-dir models

# Ridge Regression with different alphas
echo ""
echo "📊 Training Ridge Regression (alpha=0.1)..."
python train.py \
  --version v0.2-ridge-0.1 \
  --model-type ridge \
  --alpha 0.1 \
  --output-dir models

echo ""
echo "📊 Training Ridge Regression (alpha=1.0)..."
python train.py \
  --version v0.2-ridge-1.0 \
  --model-type ridge \
  --alpha 1.0 \
  --output-dir models

echo ""
echo "📊 Training Ridge Regression (alpha=10.0)..."
python train.py \
  --version v0.2-ridge-10.0 \
  --model-type ridge \
  --alpha 10.0 \
  --output-dir models

# Random Forest with different parameters
echo ""
echo "📊 Training Random Forest (n_estimators=50)..."
python train.py \
  --version v0.2-rf-50 \
  --model-type random_forest \
  --n-estimators 50 \
  --output-dir models

echo ""
echo "📊 Training Random Forest (n_estimators=100)..."
python train.py \
  --version v0.2-rf-100 \
  --model-type random_forest \
  --n-estimators 100 \
  --output-dir models

echo ""
echo "📊 Training Random Forest (n_estimators=100, max_depth=5)..."
python train.py \
  --version v0.2-rf-100-d5 \
  --model-type random_forest \
  --n-estimators 100 \
  --max-depth 5 \
  --output-dir models

# Generate comparison report
echo ""
echo "📈 Generating Model Comparison Report..."

python << 'EOF'
import json
from pathlib import Path
import pandas as pd

# Load all metrics
metrics_files = sorted(Path("models").glob("metrics_*.json"))
results = []

for metrics_file in metrics_files:
    with open(metrics_file) as f:
        metrics = json.load(f)
        
        results.append({
            "Version": metrics["version"],
            "Model": metrics["model_type"],
            "Params": str(metrics.get("model_params", {})),
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "R²": metrics["r2"],
            "Precision": metrics["calibration"]["precision"],
            "Recall": metrics["calibration"]["recall"],
            "F1": metrics["calibration"]["f1_score"]
        })

# Create DataFrame and sort by RMSE
df = pd.DataFrame(results)
df = df.sort_values("RMSE")

print("\n" + "="*100)
print("MODEL COMPARISON REPORT")
print("="*100)
print(df.to_string(index=False))
print("="*100)

# Find best model
best_idx = df["RMSE"].idxmin()
best_model = df.loc[best_idx]

print(f"\n🏆 BEST MODEL: {best_model['Version']}")
print(f"   Model Type: {best_model['Model']}")
print(f"   RMSE: {best_model['RMSE']:.4f}")
print(f"   R²: {best_model['R²']:.4f}")
print(f"   F1 Score: {best_model['F1']:.4f}")

# Calculate improvement over baseline
baseline = df[df["Version"] == "v0.1"].iloc[0]
rmse_improvement = ((baseline["RMSE"] - best_model["RMSE"]) / baseline["RMSE"]) * 100
r2_improvement = ((best_model["R²"] - baseline["R²"]) / baseline["R²"]) * 100
f1_improvement = ((best_model["F1"] - baseline["F1"]) / baseline["F1"]) * 100

print(f"\n📊 IMPROVEMENT OVER BASELINE:")
print(f"   RMSE: {rmse_improvement:+.2f}%")
print(f"   R²: {r2_improvement:+.2f}%")
print(f"   F1 Score: {f1_improvement:+.2f}%")

# Save report
df.to_csv("models/comparison_report.csv", index=False)
print(f"\n💾 Report saved to: models/comparison_report.csv")
EOF

echo ""
echo "✅ Iteration 2 Complete!"
echo ""
echo "Next steps:"
echo "  1. Review models/comparison_report.csv"
echo "  2. Select best model and copy to model_v0.2.pkl"
echo "  3. Update CHANGELOG.md with metrics"
echo "  4. Build Docker image with new model"
echo "  5. Commit and push to trigger CI/CD"
