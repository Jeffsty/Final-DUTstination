import pandas as pd
import ast
import json
import matplotlib.pyplot as plt
import os

# Results CSV path
csv_path = "results/f1_scoring/combined_results_first_test_only.csv"  

# Load results
df = pd.read_csv(csv_path)

# Plot ROC curves for each model
plt.figure(figsize=(8, 6))

for idx, row in df.iterrows():
    # evaluate the string representations of lists
    try:
        fpr = json.loads(row["ROC_FPRs"])
        tpr = json.loads(row["ROC_TPRs"])
        thresholds = json.loads(row["ROC_Thresholds"])
    except Exception:
        # fallback to ast if legacy
        fpr = ast.literal_eval(row["ROC_FPRs"])
        tpr = ast.literal_eval(row["ROC_TPRs"])
        thresholds = ast.literal_eval(row["ROC_Thresholds"])
    model_name = row["Model"]
    plt.plot(fpr, tpr, label=f"{model_name}")

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves for Different Models")
plt.legend(loc="lower right")
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/roc_curves.png", dpi=200)
plt.show()

metrics = ["Accuracy", "F1", "ROC_AUC", "Precision", "Recall"]
for metric in metrics:
    plt.figure(figsize=(6, 4))
    plt.bar(df["Model"], df[metric])
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.title(f"{metric} by Model")
    plt.tight_layout()
    plt.savefig(f"plots/{metric.lower()}_bar.png", dpi=200)
    plt.show()

for idx, row in df.iterrows():
    model_name = row["Model"]
    try:
        probas = json.loads(row["Probability_Scores"])
    except Exception:
        try:
            probas = ast.literal_eval(row["Probability_Scores"])
        except Exception as e:
            print(f"Could not load probabilities for {model_name}: {e}")
            continue

    plt.figure(figsize=(7, 4))
    plt.hist(probas, bins=20, range=(0, 1), alpha=0.7, color="tab:blue", edgecolor="k")
    plt.xlabel("Predicted Probability (for class 1)")
    plt.ylabel("Count")
    plt.title(f"Probability Score Distribution: {model_name}")
    plt.tight_layout()
    plt.savefig(f"plots/probability_scores_{model_name.replace(' ', '_')}.png", dpi=200)
    plt.show()

print("All plots saved in 'plots/' directory.")