import os
import pandas as pd
import numpy as np
import json
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split

# Load your processed/test data (adjust as needed)
chosen_data_set = input("1. All Tests Data\n2. First Test Only\n")
if chosen_data_set == "1":
    file_name = "consolidated_all_tests_data.csv"
    dataset_name = "all_tests"
elif chosen_data_set == "2":
    file_name = "consolidated_first_test_only_data.csv"
    dataset_name = "first_test_only"
else:
    raise ValueError("Invalid input.")
df = pd.read_csv(f'./consolidated_data/{file_name}')

df = df.drop(['Test Start Time', 'Test End Time'], axis=1)
categorical_cols = ["Result", "Failed Test Case", "Day of Week", "Month"]
numerical_cols = ["Hour"]

X = df.drop("False Positive", axis=1)
y = df["False Positive"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# You should use the same encoding/scaling as your main script here!
from sklearn.preprocessing import OneHotEncoder, StandardScaler
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])
X_train_pre = np.hstack([X_train_num, X_train_cat])
X_test_pre = np.hstack([X_test_num, X_test_cat])

# Train DummyClassifier (most_frequent strategy)
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_pre, y_train)
y_pred_dummy = dummy.predict(X_test_pre)
y_prob_dummy = dummy.predict_proba(X_test_pre)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_dummy)
cm = confusion_matrix(y_test, y_pred_dummy)

# Collect metrics
res = {
    "Model": "Dummy (Most Frequent)",
    "Accuracy": accuracy_score(y_test, y_pred_dummy),
    "Precision": precision_score(y_test, y_pred_dummy, zero_division=0),
    "Recall": recall_score(y_test, y_pred_dummy, zero_division=0),
    "F1": f1_score(y_test, y_pred_dummy, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, y_prob_dummy),
    "TN": cm[0, 0],
    "FP": cm[0, 1],
    "FN": cm[1, 0],
    "TP": cm[1, 1],
    "Num_Incorrect": np.sum(y_pred_dummy != y_test),
    "Pct_Incorrect": (np.sum(y_test != y_pred_dummy) / len(y_pred_dummy)) * 100,
    "Params": str(dummy.get_params()),
    "n_iter": "-",
    "Search_Time_seconds": "-",
    "Classification_Report": classification_report(y_test, y_pred_dummy, output_dict=True),
    "ROC_FPRs": json.dumps(list(fpr)),
    "ROC_TPRs": json.dumps(list(tpr)),
    "ROC_Thresholds": json.dumps(list(thresholds)),
    "Probability_Scores": json.dumps(list(y_prob_dummy)),
}

# Save/append to your results CSV
results_csv = f"results/dummy_results_{dataset_name}.csv"
if os.path.exists(results_csv):
    df_results = pd.read_csv(results_csv)
    # Prevent duplicate Dummy rows
    if (df_results["Model"] == "Dummy (Most Frequent)").any():
        print("Dummy baseline already in results CSV.")
    else:
        df_results = pd.concat([df_results, pd.DataFrame([res])], ignore_index=True)
        df_results.to_csv(results_csv, index=False)
        print("Dummy baseline added to results CSV.")
else:
    pd.DataFrame([res]).to_csv(results_csv, index=False)
    print("Results CSV created with Dummy baseline.")

print("Dummy baseline metrics:")
for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
    print(f"{k}: {res[k]:.3f}")