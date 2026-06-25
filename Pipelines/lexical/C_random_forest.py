
# PIPELINE C: TF-IDF + RANDOM FOREST

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_and_save import evaluate_and_save
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


os.makedirs("results", exist_ok=True)

print("\n===== PIPELINE C: TF-IDF + RANDOM FOREST =====")

df = pd.read_csv('training_final_dataset.csv')

def preprocess(text):
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_text'] = df['text'].astype(str).apply(preprocess)

X = df['clean_text']
y = df['label'].values

print(f"Dataset Size: {len(df)}")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'tfidf__max_features': [2000, 5000, 8000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'rf__n_estimators': [None, 100, 200],
    'rf__max_depth': [10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=kfold,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print(f"\nBest CV F1 (macro): {grid.best_score_:.4f}")

y_pred = cross_val_predict(best_model, X, y, cv=kfold)

evaluate_and_save(
    "PIPELINE_C_TFIDF_RF",
    grid,
    best_model,
    X,
    y,
    y_pred,
    kfold
)