import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv("sonnet_zero_shot.csv")

# =========================================================
# STANDARDIZE LABELS
# =========================================================

df['Label'] = (
    df['Label']
    .astype(str)
    .str.lower()
    .str.strip()
)

df['llm'] = (
    df['llm']
    .astype(str)
    .str.lower()
    .str.strip()
)

# =========================================================
# TRUE & PREDICTED LABELS
# =========================================================

y_true = df['Label']
y_pred = df['llm']

# =========================================================
# CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted'
    )
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n======================================")
print("Claude Sonnet 4.6 Performance Metrics")
print("======================================")

print(f"\nAccuracy            : {accuracy * 100:.2f}%")
print(f"Weighted Precision  : {precision * 100:.2f}%")
print(f"Weighted Recall     : {recall * 100:.2f}%")
print(f"Weighted F1-Score   : {f1 * 100:.2f}%")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n======================================")
print("Classification Report")
print("======================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)