"""
===========================================================
TASK 4: DISEASE PREDICTION FROM MEDICAL DATA
===========================================================

Project:
Disease Prediction using Machine Learning

Algorithms:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost

Dataset:
Breast Cancer Wisconsin Diagnostic Dataset

Objective:
Predict whether a tumor is malignant or benign using
medical diagnostic features.

NOTE:
This project is for educational and machine-learning
demonstration purposes only. It is NOT a medical diagnosis
system.
===========================================================
"""

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

warnings.filterwarnings("ignore")


# ==========================================================
# 2. CREATE PROJECT DIRECTORIES
# ==========================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

print("\n" + "=" * 65)
print("DISEASE PREDICTION FROM MEDICAL DATA")
print("=" * 65)

print("\nLoading Breast Cancer Wisconsin Dataset...")

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Original sklearn target:
# 0 = malignant
# 1 = benign
#
# We convert it to:
# 1 = malignant
# 0 = benign

y = pd.Series(
    np.where(data.target == 0, 1, 0),
    name="disease"
)

print("\nDataset loaded successfully!")

print(f"Number of records : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")


# ==========================================================
# 4. CREATE COMPLETE DATAFRAME
# ==========================================================

df = X.copy()
df["disease"] = y

print("\nFirst five records:")
print(df.head())


# ==========================================================
# 5. DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 65)
print("DATASET INFORMATION")
print("=" * 65)

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nTarget distribution:")
print(
    df["disease"]
    .value_counts()
    .rename(index={
        0: "Benign",
        1: "Malignant"
    })
)


# ==========================================================
# 6. BASIC STATISTICAL ANALYSIS
# ==========================================================

print("\n" + "=" * 65)
print("STATISTICAL SUMMARY")
print("=" * 65)

print(df.describe().T.head(10))


# ==========================================================
# 7. VISUALIZE CLASS DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    x=df["disease"].map({
        0: "Benign",
        1: "Malignant"
    })
)

plt.title("Disease Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
plt.tight_layout()

plt.savefig(
    "outputs/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 8. CORRELATION HEATMAP
# ==========================================================

# Select a subset of important features for readability
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "disease"
]

plt.figure(figsize=(12, 9))

sns.heatmap(
    df[selected_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Medical Features")
plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 9. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 65)
print("TRAIN-TEST SPLIT")
print("=" * 65)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ==========================================================
# 10. DEFINE MACHINE LEARNING MODELS
# ==========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                probability=True,
                kernel="rbf",
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ==========================================================
# 11. TRAIN MODELS
# ==========================================================

print("\n" + "=" * 65)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 65)

results = []

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    trained_models[name] = model

    print(f"{name} trained successfully!")


# ==========================================================
# 12. MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================================
# 13. SAVE MODEL COMPARISON
# ==========================================================

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ==========================================================
# 14. MODEL COMPARISON VISUALIZATION
# ==========================================================

plot_df = results_df.set_index("Model")

plot_df[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
].plot(
    kind="bar",
    figsize=(13, 7)
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Algorithm")
plt.ylim(0.80, 1.02)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 15. FIND BEST MODEL
# ==========================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 65)
print("BEST MODEL")
print("=" * 65)

print(f"Best performing model: {best_model_name}")

best_score = results_df.iloc[0]["F1 Score"]

print(f"F1 Score: {best_score:.4f}")


# ==========================================================
# 16. CLASSIFICATION REPORT
# ==========================================================

best_predictions = best_model.predict(X_test)

print("\n" + "=" * 65)
print(f"CLASSIFICATION REPORT — {best_model_name}")
print("=" * 65)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )
)


# ==========================================================
# 17. CONFUSION MATRIX
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.ravel()

for index, (name, model) in enumerate(
    trained_models.items()
):

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[index],
        xticklabels=[
            "Benign",
            "Malignant"
        ],
        yticklabels=[
            "Benign",
            "Malignant"
        ]
    )

    axes[index].set_title(
        f"{name} - Confusion Matrix"
    )

    axes[index].set_xlabel(
        "Predicted"
    )

    axes[index].set_ylabel(
        "Actual"
    )

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 18. ROC CURVES
# ==========================================================

plt.figure(figsize=(10, 7))

for name, model in trained_models.items():

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc_score:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 19. FEATURE IMPORTANCE
# ==========================================================

if best_model_name in [
    "Random Forest",
    "XGBoost"
]:

    importances = best_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=feature_importance,
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"Top Medical Features — {best_model_name}"
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Medical Feature")

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nTop important medical features:")

    print(
        feature_importance.to_string(
            index=False
        )
    )


# ==========================================================
# 20. SAVE BEST MODEL
# ==========================================================

model_path = (
    "models/disease_prediction_model.pkl"
)

joblib.dump(
    {
        "model": best_model,
        "features": list(X.columns),
        "target_mapping": {
            0: "Benign",
            1: "Malignant"
        }
    },
    model_path
)

print("\nBest model saved successfully!")

print(
    f"Model location: {model_path}"
)


# ==========================================================
# 21. EXAMPLE PATIENT PREDICTION
# ==========================================================

print("\n" + "=" * 65)
print("EXAMPLE PATIENT PREDICTION")
print("=" * 65)

# Select one patient from the test dataset
sample_patient = X_test.iloc[[0]]

actual_value = y_test.iloc[0]

prediction = best_model.predict(
    sample_patient
)[0]

probability = best_model.predict_proba(
    sample_patient
)[0][1]

actual_label = (
    "Malignant"
    if actual_value == 1
    else "Benign"
)

predicted_label = (
    "Malignant"
    if prediction == 1
    else "Benign"
)

print(f"\nActual diagnosis    : {actual_label}")
print(f"Predicted diagnosis : {predicted_label}")
print(
    f"Malignant probability: {probability * 100:.2f}%"
)


# ==========================================================
# 22. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nModels evaluated:")

for model_name in models:
    print(f"  ✓ {model_name}")

print("\nBest model:")
print(f"  ✓ {best_model_name}")

print("\nGenerated files:")
print("  ✓ outputs/class_distribution.png")
print("  ✓ outputs/correlation_heatmap.png")
print("  ✓ outputs/model_comparison.png")
print("  ✓ outputs/confusion_matrices.png")
print("  ✓ outputs/roc_curves.png")
print("  ✓ outputs/model_comparison.csv")
print("  ✓ outputs/feature_importance.png")
print("  ✓ models/disease_prediction_model.pkl")

print("\nDisease prediction project finished!")