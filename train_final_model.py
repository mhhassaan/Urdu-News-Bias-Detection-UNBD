"""Rebuild and evaluate a deployment-compatible Pipeline I artifact set.

This script intentionally does not preserve the reported 83.48% claim. It
regenerates deterministic LaBSE embeddings, evaluates the exact scaled deployed
pipeline with nested cross-validation, and writes fresh measured metrics.
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


DATA_PATH = Path("training_final_dataset.csv")
EMBEDDING_PATH = Path("labse_embeddings_deterministic.npy")
EMBEDDING_METADATA_PATH = Path("labse_embeddings_deterministic.json")
MODEL_DIR = Path("backend/model")

TOKENIZER_NAME = "setu4993/LaBSE"
MODEL_NAME = "sentence-transformers/LaBSE"
MAX_LENGTH = 128
RANDOM_STATE = 42


def preprocess(text):
    text = re.sub(r"[a-zA-Z]", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


def text_digest(texts):
    payload = "\n".join(texts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def encode_labse(texts, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoding deterministic LaBSE features on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
        embeddings.append(pooled.cpu().numpy())

        if start == 0 or (start // batch_size) % 25 == 0:
            print(f"Encoded {min(start + len(batch), len(texts))}/{len(texts)}")

    return np.vstack(embeddings).astype(np.float64)


def load_or_create_embeddings(texts, batch_size, force):
    digest = text_digest(texts)
    if not force and EMBEDDING_PATH.exists() and EMBEDDING_METADATA_PATH.exists():
        metadata = json.loads(EMBEDDING_METADATA_PATH.read_text(encoding="utf-8"))
        embeddings = np.load(EMBEDDING_PATH)
        if (
            metadata.get("text_sha256") == digest
            and embeddings.shape == (len(texts), 768)
        ):
            print("Using validated deterministic LaBSE embedding cache.")
            return embeddings

    embeddings = encode_labse(texts, batch_size)
    np.save(EMBEDDING_PATH, embeddings)
    EMBEDDING_METADATA_PATH.write_text(
        json.dumps(
            {
                "text_sha256": digest,
                "rows": len(texts),
                "dimensions": int(embeddings.shape[1]),
                "tokenizer": TOKENIZER_NAME,
                "model": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "pooling": "attention_mask_mean",
                "model_eval": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings


def nested_cross_validate(texts, embeddings, labels):
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    predictions = np.zeros(len(labels), dtype=int)
    probabilities = np.zeros(len(labels), dtype=float)
    best_params = []

    for fold, (train_index, test_index) in enumerate(
        outer_cv.split(texts, labels),
        start=1,
    ):
        print(f"Outer fold {fold}/5")
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        train_tfidf = tfidf.fit_transform(texts[train_index]).toarray()
        test_tfidf = tfidf.transform(texts[test_index]).toarray()

        train_features = np.hstack(
            (train_tfidf, embeddings[train_index])
        )
        test_features = np.hstack((test_tfidf, embeddings[test_index]))

        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=5000,
                        penalty="l2",
                    ),
                ),
            ]
        )
        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE + fold,
        )
        search = GridSearchCV(
            estimator,
            {
                "lr__C": [0.01, 0.1, 1, 10],
                "lr__solver": ["liblinear", "lbfgs"],
            },
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
        )
        search.fit(train_features, labels[train_index])

        predictions[test_index] = search.predict(test_features)
        class_index = list(search.classes_).index(1)
        probabilities[test_index] = search.predict_proba(test_features)[
            :, class_index
        ]
        best_params.append(search.best_params_)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
        ),
        "outer_folds": 5,
        "inner_folds": 3,
        "out_of_fold_predictions": True,
    }
    return metrics, best_params, probabilities


def most_common_parameters(parameter_sets):
    normalized = [tuple(sorted(item.items())) for item in parameter_sets]
    return dict(Counter(normalized).most_common(1)[0][0])


def fit_and_save(texts, embeddings, labels, parameters, metrics):
    print("Fitting final deployment artifacts on all labeled samples...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(texts).toarray()
    raw_features = np.hstack((tfidf_features, embeddings))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features)
    model = LogisticRegression(
        max_iter=5000,
        C=float(parameters["lr__C"]),
        solver=str(parameters["lr__solver"]),
        penalty="l2",
    )
    model.fit(scaled_features, labels)

    rng = np.random.default_rng(RANDOM_STATE)
    background_indices = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        background_indices.extend(
            rng.choice(candidates, size=min(50, len(candidates)), replace=False)
        )
    background_indices = np.asarray(background_indices)
    background = scaled_features[background_indices]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(model, MODEL_DIR / "lr_model.pkl")
    np.save(MODEL_DIR / "x_background.npy", background)

    metadata = {
        "artifact_version": "2.0.0",
        "runtime_compatibility": "exact",
        "labse_tokenizer": TOKENIZER_NAME,
        "labse_model": MODEL_NAME,
        "labse_weights_note": (
            "sentence-transformers/LaBSE and setu4993/LaBSE cached model "
            "weights were SHA-256 identical during the 2026-06-19 audit."
        ),
        "labse_max_length": MAX_LENGTH,
        "labse_pooling": "attention_mask_mean",
        "labse_model_eval": True,
        "tfidf_max_features": 3000,
        "tfidf_ngram_range": [1, 2],
        "scaler": "StandardScaler",
        "classifier": "LogisticRegression",
        "classifier_parameters": parameters,
        "evaluation": metrics,
        "evaluation_unit": "article",
        "sentence_level_metrics_available": False,
        "probability_calibrated": False,
        "training_rows": int(len(labels)),
        "training_text_sha256": text_digest(texts.tolist()),
        "background_rows": int(len(background)),
        "background_sampling": "stratified_random",
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Regenerate deterministic embeddings even when a valid cache exists.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(DATA_PATH)
    texts = frame["text"].astype(str).map(preprocess).to_numpy()
    labels = frame["label"].to_numpy()
    embeddings = load_or_create_embeddings(
        texts.tolist(),
        batch_size=args.batch_size,
        force=args.force_embeddings,
    )

    metrics, parameter_sets, _ = nested_cross_validate(
        texts,
        embeddings,
        labels,
    )
    parameters = most_common_parameters(parameter_sets)
    fit_and_save(texts, embeddings, labels, parameters, metrics)


if __name__ == "__main__":
    main()
