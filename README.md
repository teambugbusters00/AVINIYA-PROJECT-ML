# Aviniya ML Baseline

A reproducible tabular classification pipeline using scikit-learn.

```bash
pip install -r requirements.txt
python train.py --data data.csv --target target
```

Input must be a CSV with a target column. The script handles numeric and categorical features, trains a random forest pipeline, prints held-out metrics, and saves model.joblib. Check leakage, class imbalance, and external validation before relying on results.