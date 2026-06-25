# ===============================
# PIPELINE E: LaBSE + SVM
# ===============================

import os
import re
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from evaluate_and_save import evaluate_and_save

print("\n===== PIPELINE E: LaBSE + SVM =====")

df = pd.read_csv('../training_final_dataset.csv')

def preprocess(text):
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_text'] = df['text'].astype(str).apply(preprocess)

texts = df['clean_text'].tolist()
y = df['label'].values

print(f"Dataset Size: {len(df)}")

embedding_path = "../labse_embeddings.npy"

if os.path.exists(embedding_path):
    print("\nLoading saved LaBSE embeddings...")
    X = np.load(embedding_path)

else:
    print("\nNo saved embeddings found. Extracting LaBSE embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("setu4993/LaBSE")
    model = AutoModel.from_pretrained("setu4993/LaBSE").to(device)

    def get_embeddings(text_list, batch_size=32):
        embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)

        return np.vstack(embeddings)

    X = get_embeddings(texts)

    # Save embeddings
    np.save(embedding_path, X)
    print(f"Embeddings saved to {embedding_path}")

print(f"Embedding Shape: {X.shape}")

param_grid = {
    'C': [0.01, 0.1, 1, 10]
}

print("\nParameter Grid:")
print(param_grid)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    LinearSVC(max_iter=10000),
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
    "PIPELINE_E_LABSE_SVM",
    grid,
    best_model,
    X,
    y,
    y_pred,
    kfold
)