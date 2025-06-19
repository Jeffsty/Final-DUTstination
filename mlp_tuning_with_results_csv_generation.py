import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
import time
import json

def print_tpr_fpr_thresholds(y_true, y_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    print(f"\n--- ROC Curve for {model_name} ---")
    print("Threshold\tFPR\tTPR")
    for thr, fp, tp in zip(thresholds, fpr, tpr):
        print(f"{thr:.3f}\t{fp:.3f}\t{tp:.3f}")

# Load the CSV file
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

print("dropping start and end time columns")

# Drop start and end time columns
df = df.drop(['Test Start Time', 'Test End Time'], axis=1)
print(f"Original DataFrame loaded. Shape: {df.shape}")

print("specifying numerical and categorical features")

# specify numerical and categorical features
categorical_cols = ["Result", "Failed Test Case", "Day of Week", "Month"]
numerical_cols = ["Hour"]

print("splitting features and target")

# Split features and target
X = df.drop("False Positive", axis=1)
y = df["False Positive"]

print("splitting data in train and test sets 80/20")

# Split Data into Train and Test Sets 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("applying onw hot encoding to categorical features")

# Apply One-Hot Encoding to Categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

print("standardising numerical columns with scaling")

# Standardising numerical columns with Scaling
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])

print("combining encoded and scaled features")

# Combine encoded and scaled features
X_train_pre = np.hstack([X_train_num, X_train_cat])
X_test_pre = np.hstack([X_test_num, X_test_cat])

print("applygin SMOTE")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_pre, y_train)

print("cv splitting")

# CV Splitter
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("preparing to collect results")

# Prepare to collect results
results = []

def collect_metrics(name, y_true, y_pred, y_prob, params, n_iter=None, search_time=None):
    cr = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    res = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
        "Num_Incorrect": np.sum(y_pred != y_true),
        "Pct_Incorrect": (np.sum(y_true != y_pred) / len(y_pred)) * 100,
        "Params": params,
        "n_iter": n_iter,
        "Search_Time_seconds": search_time,
        "Classification_Report": cr,
        "ROC_FPRs": json.dumps(list(fpr)),
        "ROC_TPRs": json.dumps(list(tpr)),
        "ROC_Thresholds": json.dumps(list(thresholds)),
        "Probability_Scores": json.dumps(list(y_prob)),
    }
    return res

print("running mlp: default")

# MultiLayer Perceptron: default hyperparameters
mlp_default = MLPClassifier(random_state=42, max_iter=1000)
mlp_default_start = time.time()
mlp_default.fit(X_train_bal, y_train_bal)
mlp_default_end = time.time()
default_time = mlp_default_end - mlp_default_start
y_pred_mlp_default = mlp_default.predict(X_test_pre)
y_prob_mlp_default = mlp_default.predict_proba(X_test_pre)[:, 1]
results.append(collect_metrics(
    "MLP (Default)", y_test, y_pred_mlp_default, y_prob_mlp_default, mlp_default.get_params(), mlp_default.n_iter_, search_time=default_time
))

print("running mlp: randomisedsearchcv")

# MultiLayer Perceptron: RandomizedSearchCV hyperparameters
mlp_param_dist = {
    "hidden_layer_sizes": [(10,), (20,), (50,), 
                           (100,), (20, 10), (50, 25), 
                           (50, 50), (100, 50), (50, 50, 25)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
}
search_mlp_rand = RandomizedSearchCV(
    MLPClassifier(random_state=42, max_iter=1000),
    param_distributions=mlp_param_dist,
    n_iter=20,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
mlp_rand_start = time.time()
search_mlp_rand.fit(X_train_bal, y_train_bal)
mlp_rand_end = time.time()
rand_time = mlp_rand_end - mlp_rand_start
mlp_tuned_rand = search_mlp_rand.best_estimator_
y_pred_mlp_tuned_rand = mlp_tuned_rand.predict(X_test_pre)
y_prob_mlp_tuned_rand = mlp_tuned_rand.predict_proba(X_test_pre)[:, 1]
results.append(collect_metrics(
    "MLP (RandomizedSearchCV)", y_test, y_pred_mlp_tuned_rand, y_prob_mlp_tuned_rand, search_mlp_rand.best_params_, mlp_tuned_rand.n_iter_, search_time=rand_time
))
print(f"RandomizedSearchCV took {(mlp_rand_end - mlp_rand_start):.2f} seconds")

print("running mlp: gridsearchcv")

# MultiLayer Perceptron: GridSearchCV hyperparameters
search_mlp_grid = GridSearchCV(
    MLPClassifier(random_state=42, max_iter=1000),
    param_grid=mlp_param_dist,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
mlp_grid_start = time.time()
search_mlp_grid.fit(X_train_bal, y_train_bal)
mlp_grid_end = time.time()
grid_time = mlp_grid_end - mlp_grid_start
mlp_tuned_grid = search_mlp_grid.best_estimator_
y_pred_mlp_tuned_grid = mlp_tuned_grid.predict(X_test_pre)
y_prob_mlp_tuned_grid = mlp_tuned_grid.predict_proba(X_test_pre)[:, 1]
results.append(collect_metrics(
    "MLP (GridSearchCV)", y_test, y_pred_mlp_tuned_grid, y_prob_mlp_tuned_grid, search_mlp_grid.best_params_, mlp_tuned_grid.n_iter_, search_time=grid_time
))
print(f"GridSearchCV took {(mlp_grid_end - mlp_grid_start):.2f} seconds")

print("saving results to csv")

# Save results to CSV
os.makedirs("results", exist_ok=True)
csv_filename = f"results/f1_scoring/mlp_results_{dataset_name}.csv"

# Flatten classification report for saving
flat_results = []
for res in results:
    flat = res.copy()
    cr = flat.pop("Classification_Report")
    # Flatten main class metrics (just for 0/1, micro/macro/weighted avg)
    for key, val in cr.items():
        if isinstance(val, dict):
            for metric, score in val.items():
                flat[f"{key}_{metric}"] = score
        else:
            flat[f"classification_{key}"] = val
    flat_results.append(flat)

results_df = pd.DataFrame(flat_results)
results_df.to_csv(csv_filename, index=False)
print(f"\nAll metrics and results saved to: {csv_filename}")

# Optionally, still print to screen
for res in results:
    print(f"\n=== {res['Model']} ===")
    print(f"Accuracy:  {res['Accuracy']:.3f}")
    print(f"Precision: {res['Precision']:.3f}")
    print(f"Recall:    {res['Recall']:.3f}")
    print(f"F1:        {res['F1']:.3f}")
    print(f"ROC AUC:   {res['ROC_AUC']:.3f}")
    print(f"Params:    {res['Params']}")
    print(f"n_iter:    {res['n_iter']}")
    print(f"Num Incorrect: {res['Num_Incorrect']}")
    print(f"Pct Incorrect: {res['Pct_Incorrect']:.2f}")
    print(f"TN: {res['TN']} | FP: {res['FP']} | FN: {res['FN']} | TP: {res['TP']}")
    # Print main classification report as string too
    print(pd.DataFrame(res['Classification_Report']))