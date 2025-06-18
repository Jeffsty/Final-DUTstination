import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load the CSV file
chosen_data_set = input("1. All Tests Data\n"
"2. First Test Only\n")
if chosen_data_set == "1":
    file_name = "consolidated_all_tests_data.csv"
elif chosen_data_set == "2":
    file_name = "consolidated_first_test_only_data.csv"
df = pd.read_csv(f'./consolidated_data/{file_name}')

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

# CV Splitter
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Random Forest: default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train_bal, y_train_bal)
y_pred_rf_default = rf_default.predict(X_test_pre)
y_prob_rf_default = rf_default.predict_proba(X_test_pre)[:, 1]

# Random Forest: RandomizedSearchCV hyperparameters
rf_param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}
search_rf_rand = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=cv,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)
search_rf_rand.fit(X_train_bal, y_train_bal)
rf_tuned_rand = search_rf_rand.best_estimator_
y_pred_rf_tuned_rand = rf_tuned_rand.predict(X_test_pre)
y_prob_rf_tuned_rand = rf_tuned_rand.predict_proba(X_test_pre)[:, 1]

# Random Forest: GridSearchCV hyperparameters
search_rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_param_dist,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)
search_rf_grid.fit(X_train_bal, y_train_bal)
rf_tuned_grid = search_rf_grid.best_estimator_
y_pred_rf_tuned_grid = rf_tuned_grid.predict(X_test_pre)
y_prob_rf_tuned_grid = rf_tuned_grid.predict_proba(X_test_pre)[:, 1]

# Metrics function
def print_metrics(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred):.3f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, y_prob):.3f}")
    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"Number incorrect: {np.sum(y_pred != y_true)}")
    print(f"Percent Incorrect: {(np.sum(y_true != y_pred) / len(y_pred)) * 100}")
    print(f"True Negatives (TN): {confusion_matrix(y_true, y_pred)[0, 0]}")
    print(f"False Positives (FP): {confusion_matrix(y_true, y_pred)[0, 1]}")
    print(f"False Negatives (FN): {confusion_matrix(y_true, y_pred)[1, 0]}")
    print(f"True Positives (TP): {confusion_matrix(y_true, y_pred)[1, 1]}")

# Print results for all models
print_metrics("Random Forest (Default)", y_test, y_pred_rf_default, y_prob_rf_default)
print_metrics("Random Forest (RandomizedSearchCV)", y_test, y_pred_rf_tuned_rand, y_prob_rf_tuned_rand)
print_metrics("Random Forest (GridSearchCV)", y_test, y_pred_rf_tuned_grid, y_prob_rf_tuned_grid)

print("\nBest RF hyperparameters (RandomizedSearchCV):", search_rf_rand.best_params_)
print("Best RF hyperparameters (GridSearchCV):", search_rf_grid.best_params_)