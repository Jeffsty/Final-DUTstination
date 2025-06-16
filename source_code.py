import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the CSV file
chosen_data_set = input("1. All Tests Data\n"
"2. First Test Only\n")
if chosen_data_set == "1":
    file_name = "Consolidated_All_Tests_Data.csv"
elif chosen_data_set == "2":
    file_name = "Consolidated_First_Test_Only_Data.csv"
df = pd.read_csv(f'./Consolidated_Data/{file_name}')

# Drop start and end time columns
df = df.drop(['Test Start Time', 'Test End Time'], axis=1)

print(f"Original DataFrame loaded. Shape: {df.shape}")

# specify numerical and categorical features
categorical_cols = ["Result", "Failed Test Case", "Day of Week", "Month"]
numerical_cols = ["Hour"]

# Split features and target
X = df.drop("False Positive", axis=1)
y = df["False Positive"]

# Split Data into Train and Test Sets 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply One-Hot Encoding to Categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

# Standardising numerical columns with Scaling
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])

# Combine encoded and scaled features
X_train_pre = np.hstack([X_train_num, X_train_cat])
X_test_pre = np.hstack([X_test_num, X_test_cat])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_pre, y_train)

# 10. RandomizedSearchCV with 10-fold cross-validation
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}


# anything here down needs work

# 1. Default random forestModel
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train_bal, y_train_bal)
y_pred_default = rf_default.predict(X_test_pre)
y_prob_default = rf_default.predict_proba(X_test_pre)[:, 1]

# 11. Tuned model using RandomizedSearchCV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    rf_default,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
search.fit(X_train_bal, y_train_bal)
rf_tuned = search.best_estimator_
y_pred_tuned = rf_tuned.predict(X_test_pre)
y_prob_tuned = rf_tuned.predict_proba(X_test_pre)[:, 1]

print("Best hyperparameters:", search.best_params_)
print("Best cross-validation accuracy:", search.best_score_)

results = {
    "Default": [
        accuracy_score(y_test, y_pred_default),
        precision_score(y_test, y_pred_default),
        recall_score(y_test, y_pred_default),
        f1_score(y_test, y_pred_default),
        roc_auc_score(y_test, y_prob_default),
        confusion_matrix(y_test, y_pred_default),
        np.sum(y_test != y_pred_default),
        (np.sum(y_test != y_pred_default) / len(y_pred_default)) * 100
    ],
    "Tuned": [
        accuracy_score(y_test, y_pred_tuned),
        precision_score(y_test, y_pred_tuned),
        recall_score(y_test, y_pred_tuned),
        f1_score(y_test, y_pred_tuned),
        roc_auc_score(y_test, y_prob_tuned),
        confusion_matrix(y_test, y_pred_tuned),
        np.sum(y_test != y_pred_tuned),
        (np.sum(y_test != y_pred_tuned) / len(y_pred_tuned)) * 100
    ]
}

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Confusion Matrix", "Incorrect Predictions", "Incorrect Predictions %"]
results_df = pd.DataFrame(results, index=metrics)
print(results_df)

incorrect_predictions_default = np.sum(y_test != y_pred_default)
total_predictions = len(y_pred_default)
print(f"\nNumber of Incorrect Predictions: {incorrect_predictions_default} out of {total_predictions} total predictions.")
print(f"This is {incorrect_predictions_default / total_predictions:.2%} of the test set.")
print(total_predictions)

incorrect_predictions_tuned = np.sum(y_test != y_pred_tuned)
print(f"\nNumber of Incorrect Predictions: {incorrect_predictions_tuned} out of {total_predictions} total predictions.")
print(f"This is {incorrect_predictions_tuned / total_predictions:.2%} of the test set.")
print(total_predictions)


"""
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# confusion matrix stuff

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

 # --- NEW: Calculate and print number of incorrect predictions ---
incorrect_predictions = np.sum(y_test != y_pred)
total_predictions = len(y_pred)
print(f"\nNumber of Incorrect Predictions: {incorrect_predictions} out of {total_predictions} total predictions.")
print(f"This is {incorrect_predictions / total_predictions:.2%} of the test set.")

#print("Classification report:\n", classification_report(y_test, y_pred))
"""