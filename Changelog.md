Changelog
All notable changes to the Diabetes Progression Prediction Service will be documented in this file.
[v0.2] - Iteration 2 (Planned)
Goals

Improve baseline model performance
Add model calibration for high-risk classification
Implement feature selection/engineering

Changes

Model: Switch from Linear Regression to Ridge Regression or Random Forest
Preprocessing: Enhanced feature scaling and selection
Risk Calibration: Add precision/recall metrics for high-risk threshold classification
API: Add high_risk boolean flag in prediction response

Expected Metrics (to be measured)

RMSE improvement: TBD
MAE improvement: TBD
R² improvement: TBD
Risk Classification (at 75th percentile threshold):

Precision: TBD
Recall: TBD
F1 Score: TBD



Model Comparison
Ridge Regression (alpha=1.0)
bashpython train.py --version v0.2 --model-type ridge --alpha 1.0
Expected improvements:

Better regularization for small sample sizes
Reduced overfitting
More stable predictions

Random Forest (n_estimators=100)
bashpython train.py --version v0.2 --model-type random_forest --n-estimators 100
Expected improvements:

Capture non-linear relationships
Automatic feature importance
Better handling of outliers


[v0.1] - 2024-10-16 - Iteration 1 (Baseline)
Added

Training Pipeline (train.py):

Loads scikit-learn diabetes dataset
StandardScaler for feature normalization
Linear Regression baseline model
Train/test split with reproducible random state
Comprehensive metrics: RMSE, MAE, R²
Risk calibration metrics: Precision, Recall, F1 at high-risk threshold
Model and metrics persistence


API Service (app.py):

FastAPI web service
/predict endpoint for batch predictions
/predict/single endpoint for single patient
/health endpoint for monitoring
JSON error handling for invalid inputs
Risk level classification (LOW/MEDIUM/HIGH)
Predictions sorted by risk score (descending) for triage


Docker:

Multi-stage Dockerfile for minimal image size
Self-contained image with


