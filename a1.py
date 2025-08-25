import pandas as pd
import numpy as np
import time
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

def run_tuning(data_file):
    # Load Data 
    df = pd.read_csv(data_file).iloc[:150]
    X, y = df.drop("LABEL", axis=1), df["LABEL"]

    # Balance small classes
    counts = y.value_counts()
    keep_classes = counts[counts >= 2].index
    if len(keep_classes) < len(counts):
        mask = y.isin(keep_classes)
        X, y = X[mask], y[mask]

    #Train/Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        stratified = True
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        stratified = False

    # Configs
    search_space = {
        "Perceptron": (
            Perceptron(random_state=42),
            {
                "penalty": ["l2", "l1", None],
                "alpha": [0.0001, 0.001],
                "max_iter": [1000, 2000],
                "tol": [1e-3, 1e-4],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
                "min_samples_leaf": [1, 2],
            },
        ),
        "SVM": (
            SVC(random_state=42),
            {
                "C": [0.5, 1, 5],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
        ),
    }

    results, cv_data = {}, {}
    best_model = None
    best_acc = 0

    # Loop Models
    for name, (est, grid) in search_space.items():
        start = time.time()
        grid_search = GridSearchCV(
            est, grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        duration = time.time() - start

        preds = grid_search.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)

        results[name] = {
            "params": grid_search.best_params_,
            "cv": grid_search.best_score_,
            "test_acc": acc,
            "time": duration,
        }
        cv_data[name] = {"scores": cv_scores, "mean": cv_scores.mean(), "std": cv_scores.std()}

        if acc > best_acc:
            best_acc = acc
            best_model = {
                "name": name,
                "estimator": grid_search.best_estimator_,
                "params": grid_search.best_params_,
                "preds": preds,
                "cm": confusion_matrix(y_test, preds),
                "report": classification_report(y_test, preds, output_dict=True),
            }

    # Feature Importance (RF)
    feat_importance = None
    if "RandomForest" in results:
        rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        feat_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": rf.feature_importances_}
        ).sort_values("Importance", ascending=False)

    # Results
    return {
        "summary": {
            "samples": len(y),
            "features": X.shape[1],
            "train": len(X_train),
            "test": len(X_test),
            "class_dist": y.value_counts().to_dict(),
            "stratified": stratified,
        },
        "performance": results,
        "cv": cv_data,
        "importance": feat_importance,
        "best": best_model,
    }

out = run_tuning("DCT_mal.csv")

# print
print("\n=== DATA SUMMARY ===")
for k, v in out["summary"].items():
    print(f"{k:15}: {v}")

print("\n=== MODEL RESULTS ===")
for name, res in out["performance"].items():
    print(f"\n[{name}]")
    print(f"  Test Accuracy : {res['test_acc']:.4f}")
    print(f"  CV Mean       : {res['cv']:.4f}")
    print(f"  Train Time    : {res['time']:.2f}s")
    print(f"  Best Params   : {res['params']}")

print("\n=== CROSS VALIDATION (5-fold) ===")
for name, d in out["cv"].items():
    print(f"{name:12} -> mean={d['mean']:.4f}, std={d['std']:.4f}")
    print(f"   Scores: {[round(x,4) for x in d['scores']]}")

if out["importance"] is not None:
    print("\n=== TOP 5 FEATURES (RF) ===")
    for _, row in out["importance"].head(5).iterrows():
        print(f"{row['Feature']:<20}: {row['Importance']:.4f}")

print("\n=== BEST MODEL ===")
best = out["best"]
print(f"Chosen Model : {best['name']}")
print(f"Best Params  : {best['params']}")
print(f"Confusion Matrix:\n{best['cm']}")
print("\nClassification Report:")
for cls, m in best["report"].items():
    if isinstance(m, dict):
        print(f"  {cls}: " + ", ".join([f"{k}={v:.3f}" for k, v in m.items()]))

print("\n=== RANKINGS ===")
ranked = sorted(out["performance"].items(), key=lambda x: x[1]["test_acc"], reverse=True)
for i, (name, res) in enumerate(ranked, 1):
    print(f"{i}. {name:<12} | Acc={res['test_acc']:.4f} | CV={res['cv']:.4f} | Time={res['time']:.2f}s")

print("\n=== ANALYSIS COMPLETE ===")
