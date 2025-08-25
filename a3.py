import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    use_xgb = True
except:
    use_xgb = False

use_cat = False

# load dataset
df = pd.read_csv("DCT_mal.csv").iloc[:150].dropna()

X = df.drop("LABEL", axis=1).values
y = LabelEncoder().fit_transform(df["LABEL"].values)

X = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

def get_scores(clf):
    clf.fit(x_train, y_train)
    pred_train, pred_test = clf.predict(x_train), clf.predict(x_test)

    metrics = {
        "Train_Acc": accuracy_score(y_train, pred_train),
        "Test_Acc": accuracy_score(y_test, pred_test),
        "Train_Prec": precision_score(y_train, pred_train, average="weighted", zero_division=0),
        "Test_Prec": precision_score(y_test, pred_test, average="weighted", zero_division=0),
        "Train_Rec": recall_score(y_train, pred_train, average="weighted", zero_division=0),
        "Test_Rec": recall_score(y_test, pred_test, average="weighted", zero_division=0),
        "Train_F1": f1_score(y_train, pred_train, average="weighted", zero_division=0),
        "Test_F1": f1_score(y_test, pred_test, average="weighted", zero_division=0),
    }
    return metrics

# models used
models = {
    "SVM": SVC(kernel="linear", random_state=1),
    "DecisionTree": DecisionTreeClassifier(random_state=1),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=1),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=1),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=300, random_state=1)
}

if use_xgb:
    models["XGBoost"] = XGBClassifier(
        eval_metric="mlogloss", random_state=1, use_label_encoder=False,
        n_estimators=50, max_depth=3
    )

# run experiments
final_results = {}
for name, clf in models.items():
    final_results[name] = get_scores(clf)

# print
print("\n--- Classification Report (top 150 rows) ---\n")
for algo, res in final_results.items():
    print(f"### {algo} ###")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)
