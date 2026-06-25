
# PIPELINE L: TF-IDF + RoBERTa + SVM

import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from evaluate_and_save import evaluate_and_save

print("\n===== PIPELINE L: TF-IDF + RoBERTa + SVM =====")

df = pd.read_csv('../training_final_dataset.csv')

def preprocess(text):
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_text'] = df['text'].astype(str).apply(preprocess)

texts = df['clean_text'].values
y = df['label'].values

print(f"Dataset Size: {len(df)}")


embedding_path = "../roberta_embeddings.npy"

if os.path.exists(embedding_path):
    print("\nLoading RoBERTa embeddings...")
    X_roberta = np.load(embedding_path)
else:
    raise FileNotFoundError("Run Pipeline G first.")


param_grid = {
    'C': [0.01, 0.1, 1, 10]
}

print("\nParameter Grid:")
print(param_grid)


outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_preds = np.zeros(len(y))
best_params_list = []
scores = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(texts, y)):

    print(f"Fold {fold+1}/5")

    X_train_text = texts[train_idx]
    X_test_text = texts[test_idx]

    X_train_roberta = X_roberta[train_idx]
    X_test_roberta = X_roberta[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))

    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
    X_test_tfidf = tfidf.transform(X_test_text).toarray()

    X_train = np.hstack((X_train_tfidf, X_train_roberta))
    X_test = np.hstack((X_test_tfidf, X_test_roberta))

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        LinearSVC(max_iter=10000),
        param_grid,
        cv=inner_cv,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    all_preds[test_idx] = preds
    best_params_list.append(grid.best_params_)
    scores.append(grid.best_score_)

best_params_final = max(
    set(map(str, best_params_list)),
    key=list(map(str, best_params_list)).count
)
best_params_final = eval(best_params_final)

class DummyGrid:
    def __init__(self, best_params, best_score, param_grid):
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.param_grid = param_grid

grid_dummy = DummyGrid(
    best_params_final,
    np.mean(scores),
    param_grid
)

num_features = X_train.shape[1]

evaluate_and_save(
    "PIPELINE_L_TFIDF_ROBERTA_SVM",
    grid_dummy,
    best_model,
    None,
    y,
    all_preds,
    outer_cv,
    num_features=num_features
)