import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

DATA_PATH = "data/har.csv"
MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y_raw = df["label"]

# Label Encoding
y = LabelEncoder().fit_transform(y_raw)

# Using a 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# For AUC calculations
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        n_estimators=200,
        random_state=42
    )
}


results = []
for name, model in models.items():
    print(f"Training {name} model...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute AUC if probabilities exist
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(
            y_test_bin,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
    else:
        auc = np.nan

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)
    model_file = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_file))

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(RESULT_DIR, "metrics.csv"), index=False)

print("\nMetrics saved to results/metrics.csv")
print(metrics_df)
