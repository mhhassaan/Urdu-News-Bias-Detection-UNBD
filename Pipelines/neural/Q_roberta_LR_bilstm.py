# ===============================
# PIPELINE Q: RoBERTa + BiLSTM + LOGISTIC REGRESSION (NO LEAKAGE)
# ===============================

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

from evaluate_and_save import evaluate_and_save

print("\n===== PIPELINE Q: RoBERTa + BiLSTM + LOGISTIC REGRESSION (NO LEAKAGE) =====")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('../training_final_dataset.csv')

def preprocess(text):
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_text'] = df['text'].astype(str).apply(preprocess)

y = df['label'].values

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# LOAD EMBEDDINGS
# ===============================
X_seq = np.load("../roberta_seq_lstm_embeddings.npy")

# Normalize
X_seq = X_seq.astype(np.float32)
mean = np.mean(X_seq, axis=(0,1), keepdims=True)
std = np.std(X_seq, axis=(0,1), keepdims=True) + 1e-8
X_seq = (X_seq - mean) / std
X_seq = np.clip(X_seq, -3, 3)

# ===============================
# BiLSTM MODEL
# ===============================
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x, return_features=False):
        _, (h_n, _) = self.lstm(x)

        # concatenate forward + backward
        features = torch.cat((h_n[-2], h_n[-1]), dim=1)
        features = self.dropout(features)

        if return_features:
            return features

        return self.fc(features)

# ===============================
# CV SETUP
# ===============================
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_preds = np.zeros(len(y))

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear'],
    'penalty': ['l2']
}

best_params_list = []
scores = []

# ===============================
# OUTER LOOP
# ===============================
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_seq, y)):

    print(f"\n🔁 Fold {fold+1}/5")

    X_train_seq = X_seq[train_idx]
    X_test_seq  = X_seq[test_idx]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ===============================
    # TRAIN BiLSTM
    # ===============================
    print(f"Training BiLSTM for Fold {fold+1}...")

    model_lstm = BiLSTMModel().to(device)

    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-4)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    X_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    num_epochs = 10
    batch_size = 32

    for epoch in range(num_epochs):
        total_loss = 0
        indices = torch.randperm(len(X_tensor))

        loop = tqdm(
            range(0, len(X_tensor), batch_size),
            desc=f"Fold {fold+1} | Epoch {epoch+1}/{num_epochs}",
            leave=False
        )

        for i in loop:
            batch_idx = indices[i:i+batch_size]

            xb = X_tensor[batch_idx]
            yb = y_tensor[batch_idx]

            optimizer.zero_grad()
            logits = model_lstm(xb)

            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Fold {fold+1} | Epoch {epoch+1} Loss: {total_loss:.4f}")

    # ===============================
    # FEATURE EXTRACTION
    # ===============================
    def extract_features(X_data):
        model_lstm.eval()
        feats = []

        with torch.no_grad():
            for i in range(0, len(X_data), 32):
                xb = torch.tensor(X_data[i:i+32], dtype=torch.float32).to(device)
                f = model_lstm(xb, return_features=True)
                feats.append(f.cpu().numpy())

        return np.vstack(feats)

    X_train_feat = extract_features(X_train_seq)
    X_test_feat  = extract_features(X_test_seq)

    # ===============================
    # GRID SEARCH
    # ===============================
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        LogisticRegression(max_iter=5000),
        param_grid,
        cv=inner_cv,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train_feat, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test_feat)

    all_preds[test_idx] = preds
    best_params_list.append(grid.best_params_)
    scores.append(grid.best_score_)

# ===============================
# FINAL EVALUATION
# ===============================
class DummyGrid:
    def __init__(self, best_params, best_score, param_grid):
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.param_grid = param_grid

best_params_final = max(set(map(str, best_params_list)),
                        key=list(map(str, best_params_list)).count)
best_params_final = eval(best_params_final)

grid_dummy = DummyGrid(best_params_final, np.mean(scores), param_grid)

evaluate_and_save(
    "PIPELINE_Q_ROBERTA_BILSTM_LR",
    grid_dummy,
    best_model,
    X_train_feat,
    y,
    all_preds,
    outer_cv,
    num_features=X_train_feat.shape[1]
)