# ===============================
# PIPELINE P: LaBSE + LSTM + SVM (FIXED)
# ===============================

import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning

from evaluate_and_save import evaluate_and_save

# Suppress annoying sklearn warnings during GridSearch
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

print("\n===== PIPELINE P: LaBSE + LSTM + SVM (FIXED) =====")

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_seq_raw = np.load("../labse_seq_lstm_embeddings.npy").astype(np.float32)

# ===============================
# LSTM MODEL
# ===============================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, return_features=False):
        output, _ = self.lstm(x)
        features = torch.mean(output, dim=1)

        if return_features:
            return features

        return self.fc(features)

# ===============================
# CV SETUP
# ===============================
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_preds = np.zeros(len(y))

param_grid = {
    'C': [0.01, 0.1, 1, 10]
}

best_params_list = []
inner_cv_scores = []

# ===============================
# OUTER LOOP
# ===============================
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_seq_raw, y)):

    print(f"\n🔁 Fold {fold+1}/5")

    X_train_seq = X_seq_raw[train_idx]
    X_test_seq  = X_seq_raw[test_idx]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ===============================
    # CORRECT NORMALIZATION (Train only)
    # ===============================
    mean = np.mean(X_train_seq, axis=(0,1), keepdims=True)
    std = np.std(X_train_seq, axis=(0,1), keepdims=True) + 1e-8
    
    X_train_seq = (X_train_seq - mean) / std
    X_train_seq = np.clip(X_train_seq, -3, 3)
    
    X_test_seq = (X_test_seq - mean) / std
    X_test_seq = np.clip(X_test_seq, -3, 3)

    # ===============================
    # TRAIN LSTM
    # ===============================
    print(f"Training LSTM for Fold {fold+1}...")

    model_lstm = LSTMModel().to(device)
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
    # GRID SEARCH (SVM)
    # ===============================
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Added dual="auto" to satisfy recent scikit-learn LinearSVC requirements cleanly
    grid = GridSearchCV(
        LinearSVC(max_iter=10000, dual="auto"), 
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
    inner_cv_scores.append(grid.best_score_)

# ===============================
# FINAL EVALUATION
# ===============================

# Calculate the TRUE F1 Macro on the completely unseen outer predictions
true_f1_macro = f1_score(y, all_preds, average='macro')
print(f"\n✅ True Unbiased Outer CV F1-Macro: {true_f1_macro:.4f}")

class DummyGrid:
    def __init__(self, best_params, best_score, param_grid):
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.param_grid = param_grid

best_params_final = max(set(map(str, best_params_list)),
                        key=list(map(str, best_params_list)).count)
best_params_final = eval(best_params_final)

# Pass the true unbiased score into DummyGrid instead of the leaked inner score
grid_dummy = DummyGrid(best_params_final, true_f1_macro, param_grid)

# ===============================
# EVALUATE
# ===============================
evaluate_and_save(
    "PIPELINE_P_LABSE_LSTM_SVM",
    grid_dummy,
    best_model,
    X_train_feat,
    y,
    all_preds,
    outer_cv,
    num_features=X_train_feat.shape[1]
)