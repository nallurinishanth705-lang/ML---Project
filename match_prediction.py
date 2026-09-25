# ============================================================
# PROJECT:
# Multi-Class Match Result Prediction Using
# Multinomial Logistic Regression
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 65)
print("MULTI-CLASS MATCH RESULT PREDICTION")
print("USING MULTINOMIAL LOGISTIC REGRESSION")
print("=" * 65)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n1. Loading dataset...")

file_path = "football_match_prediction_ready.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n2. Dataset Information")
print("-" * 50)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. TARGET VARIABLE ANALYSIS
# ============================================================

print("\n3. Match Result Distribution")
print("-" * 50)

print(df["Result"].value_counts())

result_labels = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}

result_counts = df["Result"].value_counts()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=result_counts.index,
    y=result_counts.values
)

plt.title("Distribution of Match Results")
plt.xlabel("Match Result")
plt.ylabel("Number of Matches")
plt.xticks(
    ticks=[0, 1, 2],
    labels=[
        result_labels.get(x, x)
        for x in result_counts.index
    ]
)

plt.tight_layout()
plt.savefig(
    "graphs/match_result_distribution.png",
    dpi=300
)
plt.close()

print(
    "Saved: graphs/match_result_distribution.png"
)


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n4. Creating EDA graphs...")

# ------------------------------------------------------------
# Recent points comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Home_Recent_Points"],
    kde=True,
    label="Home Team",
    alpha=0.5
)

sns.histplot(
    df["Away_Recent_Points"],
    kde=True,
    label="Away Team",
    alpha=0.5
)

plt.title("Recent Form Points Distribution")
plt.xlabel("Recent Points")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(
    "graphs/recent_points_distribution.png",
    dpi=300
)
plt.close()


# ------------------------------------------------------------
# Average goals scored
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Home_Avg_Goals_Scored"],
    kde=True,
    label="Home Team",
    alpha=0.5
)

sns.histplot(
    df["Away_Avg_Goals_Scored"],
    kde=True,
    label="Away Team",
    alpha=0.5
)

plt.title("Average Goals Scored")
plt.xlabel("Average Goals")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(
    "graphs/average_goals_scored.png",
    dpi=300
)
plt.close()


# ------------------------------------------------------------
# Win rate distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Home_Win_Rate"],
    kde=True,
    label="Home Team",
    alpha=0.5
)

sns.histplot(
    df["Away_Win_Rate"],
    kde=True,
    label="Away Team",
    alpha=0.5
)

plt.title("Historical Win Rate Distribution")
plt.xlabel("Win Rate")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(
    "graphs/win_rate_distribution.png",
    dpi=300
)
plt.close()

print("EDA graphs created successfully.")


# ============================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================

print("\n5. Preparing features...")

# Target variable
y = df["Result"]

# ------------------------------------------------------------
# Numerical features
# ------------------------------------------------------------

numeric_features = [
    "Home_Recent_Points",
    "Away_Recent_Points",

    "Home_Avg_Goals_Scored",
    "Away_Avg_Goals_Scored",

    "Home_Avg_Goals_Conceded",
    "Away_Avg_Goals_Conceded",

    "Home_Win_Rate",
    "Away_Win_Rate"
]


# ------------------------------------------------------------
# Categorical features
# ------------------------------------------------------------

categorical_features = []

for column in [
    "Country",
    "League"
]:
    if column in df.columns:
        categorical_features.append(column)


print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# Create X
# ------------------------------------------------------------

selected_features = (
    numeric_features +
    categorical_features
)

X = df[selected_features]

print("\nTotal features used:",
      len(selected_features))


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n6. Splitting dataset...")

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 8. DATA PREPROCESSING
# ============================================================

print("\n7. Creating preprocessing pipeline...")


# ------------------------------------------------------------
# Numerical preprocessing
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# Categorical preprocessing
# ------------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ------------------------------------------------------------
# Combine preprocessing
# ------------------------------------------------------------

transformers = [
    (
        "num",
        numeric_transformer,
        numeric_features
    )
]

if categorical_features:
    transformers.append(
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)


# ============================================================
# 9. MULTINOMIAL LOGISTIC REGRESSION MODEL
# ============================================================

print(
    "\n8. Training Multinomial Logistic Regression..."
)

model = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)


# ------------------------------------------------------------
# Complete ML pipeline
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed successfully!")


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

print("\n9. Making predictions...")

y_pred = pipeline.predict(X_test)

print("Predictions completed!")


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

print("\n10. MODEL PERFORMANCE")
print("=" * 65)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n11. Classification Report")
print("-" * 65)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Away Win",
        "Draw",
        "Home Win"
    ],
    zero_division=0
)

print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n12. Creating Confusion Matrix...")

labels = ["A", "D", "H"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Away Win",
        "Draw",
        "Home Win"
    ],
    yticklabels=[
        "Away Win",
        "Draw",
        "Home Win"
    ]
)

plt.title(
    "Confusion Matrix - Multinomial Logistic Regression"
)

plt.xlabel("Predicted Result")
plt.ylabel("Actual Result")

plt.tight_layout()

plt.savefig(
    "graphs/confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "Saved: graphs/confusion_matrix.png"
)


# ============================================================
# 14. PERFORMANCE METRICS GRAPH
# ============================================================

print("\n13. Creating performance graph...")

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    metric_names,
    metric_values
)

plt.title(
    "Multinomial Logistic Regression Performance"
)

plt.ylabel("Score")

plt.ylim(0, 1)

for bar, value in zip(
    bars,
    metric_values
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "graphs/model_performance.png",
    dpi=300
)

plt.close()

print(
    "Saved: graphs/model_performance.png"
)


# ============================================================
# 15. SAVE PERFORMANCE RESULTS
# ============================================================

performance_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    }
)

performance_df.to_csv(
    "results/model_performance.csv",
    index=False
)


# ============================================================
# 16. SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(
    "results/classification_report.csv"
)


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

prediction_results = X_test.copy()

prediction_results[
    "Actual_Result"
] = y_test.values

prediction_results[
    "Predicted_Result"
] = y_pred

prediction_results[
    "Actual_Result"
] = prediction_results[
    "Actual_Result"
].map(result_labels)

prediction_results[
    "Predicted_Result"
] = prediction_results[
    "Predicted_Result"
].map(result_labels)


prediction_results.to_csv(
    "results/match_predictions.csv",
    index=False
)


# ============================================================
# 18. PREDICTION PROBABILITIES
# ============================================================

probabilities = pipeline.predict_proba(
    X_test
)

class_names = pipeline.named_steps[
    "classifier"
].classes_

probability_df = pd.DataFrame(
    probabilities,
    columns=[
        "Probability_" + str(c)
        for c in class_names
    ]
)

probability_df.to_csv(
    "results/prediction_probabilities.csv",
    index=False
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PROJECT EXECUTION COMPLETED")
print("=" * 65)

print(
    f"""
Model:
Multinomial Logistic Regression

Training Records:
{X_train.shape[0]}

Testing Records:
{X_test.shape[0]}

Accuracy:
{accuracy:.4f}

Precision:
{precision:.4f}

Recall:
{recall:.4f}

F1 Score:
{f1:.4f}
"""
)

print("Generated Graphs:")
print("1. match_result_distribution.png")
print("2. recent_points_distribution.png")
print("3. average_goals_scored.png")
print("4. win_rate_distribution.png")
print("5. confusion_matrix.png")
print("6. model_performance.png")

print("\nGenerated Result Files:")
print("1. model_performance.csv")
print("2. classification_report.csv")
print("3. match_predictions.csv")
print("4. prediction_probabilities.csv")

print(
    "\nAll files have been saved successfully."
)