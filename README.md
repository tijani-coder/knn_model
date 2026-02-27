# KNN Training

This repository contains `train_knn.py` which trains a K-Nearest Neighbors classifier on the dataset `cancer patient data sets.csv`.

Quick start:

1. Create a Python environment and install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run training:

```powershell
python train_knn.py
```

Outputs:
- `knn_model.joblib` — saved model, scaler, and optional label encoder
- `knn_predictions.csv` — test set predictions

Notes:
- The script attempts to auto-detect the target column (`diagnosis`, `target`, or last column).
- It encodes categorical features, fills numeric missing values with medians, scales features, and runs a grid search over K and weights.
