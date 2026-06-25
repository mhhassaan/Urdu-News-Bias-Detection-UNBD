import os
import re
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import shap
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LR_PATH = os.path.join(MODEL_DIR, "lr_model.pkl")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "x_background.npy")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

LABSE_TOKENIZER_NAME = "setu4993/LaBSE"
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
LABSE_MAX_LENGTH = 128

# These are decision/display policies, not measured sentence-level metrics.
UNBIASED_THRESHOLD = 0.40
BIASED_THRESHOLD = 0.60
TOP_FEATURE_LIMIT = 5

# Reference values calculated from training_final_dataset.csv (10,000 rows).
TRAINING_WORD_P95 = 75
TRAINING_WORD_MAX = 93
LONG_INPUT_REVIEW_MULTIPLIER = 2

URDU_SENTENCE_BOUNDARY = re.compile(r"(?<=[۔؟!?])\s+|\n+")


@dataclass(frozen=True)
class ModelRuntime:
    tfidf: Any
    scaler: Any
    model: Any
    explainer: Any
    labse_tokenizer: Any
    labse_model: Any
    labse_device: torch.device
    tfidf_feature_names: np.ndarray
    tfidf_dim: int
    biased_class_index: int
    metadata: Dict[str, Any]


def preprocess(text: str) -> str:
    """Match the cleaning used when Pipeline I artifacts were trained."""
    text = re.sub(r"[a-zA-Z]", "", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(clean_text: str) -> List[str]:
    """Split Urdu text on Urdu/English terminal punctuation and line breaks."""
    pieces = URDU_SENTENCE_BOUNDARY.split(clean_text)
    sentences = [piece.strip() for piece in pieces if piece.strip()]
    return sentences or ([clean_text] if clean_text else [])


def context_window(
    sentences: Sequence[str],
    index: int,
) -> Tuple[str, Dict[str, int]]:
    """Return previous/current/next context and the inclusive sentence range."""
    start = max(0, index - 1)
    end_exclusive = min(len(sentences), index + 2)
    return (
        " ".join(sentences[start:end_exclusive]),
        {"start": start, "end": end_exclusive - 1},
    )


def decision_from_probability(probability: float) -> Dict[str, Any]:
    """Convert biased-class probability into an explicit decision policy."""
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability >= BIASED_THRESHOLD:
        label = "biased"
        status = "confident"
    elif probability <= UNBIASED_THRESHOLD:
        label = "unbiased"
        status = "confident"
    else:
        label = "biased" if probability >= 0.5 else "unbiased"
        status = "uncertain"

    return {
        "label": label,
        "status": status,
        "is_uncertain": status == "uncertain",
        "biased_probability": probability,
        "predicted_class_probability": max(probability, 1.0 - probability),
        "margin_from_boundary": abs(probability - 0.5),
        "thresholds": {
            "unbiased_max": UNBIASED_THRESHOLD,
            "biased_min": BIASED_THRESHOLD,
        },
    }


@lru_cache(maxsize=1)
def get_runtime() -> ModelRuntime:
    """Load and validate model artifacts once per backend process."""
    print("Loading Pipeline I artifacts...")
    tfidf = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(LR_PATH)
    background = np.load(BACKGROUND_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    else:
        metadata = {
            "artifact_version": "legacy",
            "labse_pooling": "legacy_unmasked_mean",
            "runtime_compatibility": "approximate",
            "warning": (
                "Legacy training embeddings were created with dropout enabled "
                "and dynamic-padding mean pooling, so single-input inference "
                "cannot reproduce them exactly."
            ),
        }

    feature_names = tfidf.get_feature_names_out()
    tfidf_dim = len(feature_names)

    if background.ndim != 2 or background.shape[1] != model.n_features_in_:
        raise ValueError(
            "SHAP background dimensions do not match the Logistic Regression model."
        )
    if scaler.n_features_in_ != model.n_features_in_:
        raise ValueError("Scaler dimensions do not match the trained model.")

    class_indices = {label: index for index, label in enumerate(model.classes_)}
    if 1 not in class_indices:
        raise ValueError(
            f"Expected class 1 to represent biased text; found {model.classes_!r}."
        )

    print("Initializing SHAP LinearExplainer...")
    explainer = shap.LinearExplainer(model, background)

    print("Loading LaBSE with the training-time encoder and pooling...")
    labse_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # The two repositories publish byte-identical model weights. Keep the
    # training tokenizer while using the checkpoint layout that loads reliably
    # in the deployment environment.
    labse_tokenizer = AutoTokenizer.from_pretrained(LABSE_TOKENIZER_NAME)
    labse_model = AutoModel.from_pretrained(LABSE_MODEL_NAME).to(labse_device)
    labse_model.eval()

    return ModelRuntime(
        tfidf=tfidf,
        scaler=scaler,
        model=model,
        explainer=explainer,
        labse_tokenizer=labse_tokenizer,
        labse_model=labse_model,
        labse_device=labse_device,
        tfidf_feature_names=feature_names,
        tfidf_dim=tfidf_dim,
        biased_class_index=class_indices[1],
        metadata=metadata,
    )


def _encode_labse(runtime: ModelRuntime, texts: Sequence[str]) -> np.ndarray:
    """
    Reproduce the embedding code used to create labse_embeddings.npy.

    The original experiment used setu4993/LaBSE and an unmasked mean over
    last_hidden_state. Although attention-mask-aware pooling would normally be
    preferable, changing it at inference would create a different feature space.
    Regenerate embeddings and retrain before changing this implementation.
    """
    embeddings = []
    text_list = list(texts)
    for start in range(0, len(text_list), 32):
        batch = text_list[start : start + 32]
        inputs = runtime.labse_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=LABSE_MAX_LENGTH,
        ).to(runtime.labse_device)

        with torch.no_grad():
            outputs = runtime.labse_model(**inputs)

        if runtime.metadata.get("labse_pooling") == "attention_mask_mean":
            mask = inputs["attention_mask"].unsqueeze(-1).to(
                outputs.last_hidden_state.dtype
            )
            pooled = (outputs.last_hidden_state * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)

        embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings).astype(np.float64)


def _vectorize(
    runtime: ModelRuntime,
    texts: Sequence[str],
) -> Tuple[Any, np.ndarray]:
    x_tfidf_sparse = runtime.tfidf.transform(list(texts))
    x_tfidf = x_tfidf_sparse.toarray()
    x_labse = _encode_labse(runtime, texts)
    x_raw = np.hstack((x_tfidf, x_labse))

    if x_raw.shape[1] != runtime.scaler.n_features_in_:
        raise ValueError(
            f"Generated {x_raw.shape[1]} features, but the scaler expects "
            f"{runtime.scaler.n_features_in_}."
        )

    return x_tfidf_sparse, runtime.scaler.transform(x_raw)


def _biased_probabilities(runtime: ModelRuntime, x_scaled: np.ndarray) -> np.ndarray:
    probabilities = runtime.model.predict_proba(x_scaled)
    return probabilities[:, runtime.biased_class_index].astype(float)


def _normalise_shap_matrix(
    runtime: ModelRuntime,
    shap_values: Any,
    sample_count: int,
) -> np.ndarray:
    """Normalize SHAP versions to (samples, features) for biased log-odds."""
    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        values = values[
            runtime.biased_class_index if len(values) > 1 else 0
        ]

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, runtime.biased_class_index]
    elif values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
    if values.shape[0] != sample_count:
        raise ValueError(
            f"SHAP returned {values.shape[0]} rows for {sample_count} samples."
        )
    return values


def _shap_values(runtime: ModelRuntime, x_scaled: np.ndarray) -> np.ndarray:
    try:
        explanation = runtime.explainer(x_scaled)
    except TypeError:
        explanation = runtime.explainer.shap_values(x_scaled)
    return _normalise_shap_matrix(runtime, explanation, x_scaled.shape[0])


def _top_tfidf_features(
    runtime: ModelRuntime,
    x_tfidf_row: Any,
    shap_tfidf: np.ndarray,
    *,
    anchor_text: str | None = None,
    limit: int = TOP_FEATURE_LIMIT,
) -> List[Dict[str, Any]]:
    present_indices = x_tfidf_row.nonzero()[1]
    anchor_terms = (
        set(runtime.tfidf.build_analyzer()(anchor_text))
        if anchor_text
        else None
    )
    features: List[Dict[str, Any]] = []

    for index in present_indices:
        feature = str(runtime.tfidf_feature_names[index])
        if anchor_terms is not None and feature not in anchor_terms:
            continue
        contribution = float(shap_tfidf[index])
        features.append(
            {
                "feature": feature,
                "contribution": contribution,
                "direction": "biased" if contribution >= 0 else "unbiased",
            }
        )

    features.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return features[:limit]


def _component_explanation(
    runtime: ModelRuntime,
    x_tfidf_row: Any,
    shap_row: np.ndarray,
    *,
    anchor_text: str | None = None,
) -> Dict[str, Any]:
    shap_tfidf = shap_row[: runtime.tfidf_dim]
    shap_labse = shap_row[runtime.tfidf_dim :]

    lexical_signed = float(np.sum(shap_tfidf))
    semantic_signed = float(np.sum(shap_labse))

    return {
        "output_space": "log_odds_for_biased_class",
        "lexical": {
            "signed_contribution": lexical_signed,
            "magnitude": float(np.sum(np.abs(shap_tfidf))),
            "direction": "biased" if lexical_signed >= 0 else "unbiased",
            "top_features": _top_tfidf_features(
                runtime,
                x_tfidf_row,
                shap_tfidf,
                anchor_text=anchor_text,
            ),
        },
        "semantic": {
            "signed_contribution": semantic_signed,
            "magnitude": float(np.sum(np.abs(shap_labse))),
            "direction": "biased" if semantic_signed >= 0 else "unbiased",
            "dimensions": len(shap_labse),
            "note": (
                "LaBSE dimensions are aggregated only; they do not map directly "
                "to human-readable words or concepts."
            ),
        },
        "total_feature_contribution": float(np.sum(shap_row)),
    }


def _lexical_coverage(runtime: ModelRuntime, clean_text: str) -> Dict[str, Any]:
    analyzer = runtime.tfidf.build_analyzer()
    analyzed_terms = analyzer(clean_text)
    if not analyzed_terms:
        return {"known": 0, "total": 0, "ratio": 0.0}

    vocabulary = runtime.tfidf.vocabulary_
    known = sum(1 for term in analyzed_terms if term in vocabulary)
    return {
        "known": known,
        "total": len(analyzed_terms),
        "ratio": float(known / len(analyzed_terms)),
    }


def _evidence_label(
    sentence_probability: float,
    context_probability: float,
) -> str:
    """Require focal sentence and context to agree before strong highlighting."""
    if (
        context_probability >= BIASED_THRESHOLD
        and sentence_probability >= 0.55
    ):
        return "bias_evidence"
    if (
        context_probability <= UNBIASED_THRESHOLD
        and sentence_probability <= 0.45
    ):
        return "neutral_evidence"
    return "uncertain"


def _sentence_evidence(
    runtime: ModelRuntime,
    sentences: Sequence[str],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    if not sentences:
        return [], np.array([], dtype=float)

    contexts: List[str] = []
    spans: List[Dict[str, int]] = []
    for index in range(len(sentences)):
        context, span = context_window(sentences, index)
        contexts.append(context)
        spans.append(span)

    sentence_tfidf, sentence_scaled = _vectorize(runtime, sentences)
    context_tfidf, context_scaled = _vectorize(runtime, contexts)
    sentence_probabilities = _biased_probabilities(runtime, sentence_scaled)
    context_probabilities = _biased_probabilities(runtime, context_scaled)
    context_shap = _shap_values(runtime, context_scaled)

    items: List[Dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        sentence_probability = float(sentence_probabilities[index])
        context_probability = float(context_probabilities[index])
        evidence_label = _evidence_label(
            sentence_probability,
            context_probability,
        )
        explanation = _component_explanation(
            runtime,
            context_tfidf[index],
            context_shap[index],
            anchor_text=sentence,
        )
        top_features = explanation["lexical"]["top_features"]

        items.append(
            {
                "index": index,
                "sentence": sentence,
                "context": contexts[index],
                "context_span": spans[index],
                "sentence_probability": sentence_probability,
                "context_probability": context_probability,
                "score": context_probability - 0.5,
                "prediction": (
                    "biased" if context_probability >= 0.5 else "unbiased"
                ),
                "confidence": max(context_probability, 1.0 - context_probability),
                "evidence_label": evidence_label,
                "top_features": top_features,
                "top_words": [item["feature"] for item in top_features],
                "semantic_signal": explanation["semantic"],
                "explanation": explanation,
                "scope_note": (
                    "Exploratory context-window evidence. Pipeline I was "
                    "validated at article level, not sentence level."
                ),
            }
        )

    return items, context_probabilities


def _uncertainty_assessment(
    article_decision: Dict[str, Any],
    word_count: int,
    lexical_coverage: Dict[str, Any],
    context_probabilities: np.ndarray,
    artifact_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    severity = "low"
    artifact_metadata = artifact_metadata or {}

    if artifact_metadata.get("runtime_compatibility") != "exact":
        reasons.append(
            artifact_metadata.get(
                "warning",
                "The deployed feature extractor is not proven identical to the "
                "training feature extractor.",
            )
        )
        severity = "high"

    if article_decision["is_uncertain"]:
        reasons.append(
            "The biased-class probability falls inside the 0.40–0.60 "
            "indeterminate band."
        )
        severity = "high"

    if word_count > TRAINING_WORD_MAX * LONG_INPUT_REVIEW_MULTIPLIER:
        reasons.append(
            "The input is substantially longer than samples seen during "
            "training; LaBSE also truncates long inputs."
        )
        severity = "high"
    elif word_count > TRAINING_WORD_MAX:
        reasons.append(
            "The input is longer than the longest training sample and should "
            "be reviewed cautiously."
        )
        severity = "medium" if severity == "low" else severity

    if lexical_coverage["total"] and lexical_coverage["ratio"] < 0.35:
        reasons.append(
            "Less than 35% of analyzed lexical terms are represented in the "
            "fitted TF-IDF vocabulary."
        )
        severity = "medium" if severity == "low" else severity

    if context_probabilities.size >= 2:
        biased_share = float(np.mean(context_probabilities >= BIASED_THRESHOLD))
        neutral_share = float(np.mean(context_probabilities <= UNBIASED_THRESHOLD))
        dispersion = float(np.std(context_probabilities))
        if biased_share > 0.15 and neutral_share > 0.15:
            reasons.append(
                "Different parts of the article provide conflicting model "
                "evidence."
            )
            severity = "medium" if severity == "low" else severity
    else:
        biased_share = 0.0
        neutral_share = 0.0
        dispersion = 0.0

    status = (
        "uncertain"
        if article_decision["is_uncertain"]
        else "review"
        if reasons
        else "stable"
    )
    return {
        "status": status,
        "severity": severity,
        "review_recommended": bool(reasons),
        "reasons": reasons,
        "context_dispersion": dispersion,
        "biased_context_share": biased_share,
        "neutral_context_share": neutral_share,
        "probability_is_calibrated": False,
        "note": (
            "Logistic Regression probabilities are model scores and were not "
            "separately calibrated on a held-out calibration set."
        ),
    }


def predict(text: str) -> Dict[str, Any]:
    """
    Run article-level Pipeline I inference plus exploratory context evidence.

    The article prediction is produced from the complete cleaned input. Sentence
    and context-window outputs are diagnostic explanations only and must not be
    reported as sentence-level accuracy or validation results.
    """
    clean_text = preprocess(text)
    if not clean_text:
        raise ValueError("No usable Urdu text remained after preprocessing.")

    runtime = get_runtime()
    article_tfidf, article_scaled = _vectorize(runtime, [clean_text])
    article_probability = float(_biased_probabilities(runtime, article_scaled)[0])
    article_decision = decision_from_probability(article_probability)
    article_shap = _shap_values(runtime, article_scaled)[0]
    article_explanation = _component_explanation(
        runtime,
        article_tfidf[0],
        article_shap,
        anchor_text=clean_text,
    )

    sentences = split_sentences(clean_text)
    sentence_scores, context_probabilities = _sentence_evidence(runtime, sentences)
    lexical_coverage = _lexical_coverage(runtime, clean_text)
    word_count = len(clean_text.split())
    uncertainty = _uncertainty_assessment(
        article_decision,
        word_count,
        lexical_coverage,
        context_probabilities,
        runtime.metadata,
    )

    return {
        "prediction": article_decision["label"],
        "decision_status": article_decision["status"],
        "confidence": article_decision["predicted_class_probability"],
        "biased_probability": article_probability,
        "decision": article_decision,
        "uncertainty": uncertainty,
        "article_explanation": article_explanation,
        "sentence_scores": sentence_scores,
        "input_profile": {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "lexical_coverage": lexical_coverage,
            "training_reference": {
                "samples": 10000,
                "word_count_p95": TRAINING_WORD_P95,
                "word_count_max": TRAINING_WORD_MAX,
            },
        },
        "evaluation_scope": {
            "reported_cross_validation_accuracy": runtime.metadata.get(
                "evaluation", {}
            ).get("accuracy"),
            "reported_macro_f1": runtime.metadata.get("evaluation", {}).get(
                "macro_f1"
            ),
            "validated_unit": runtime.metadata.get(
                "evaluation_unit", "article"
            ),
            "sentence_level_metrics_available": runtime.metadata.get(
                "sentence_level_metrics_available", False
            ),
            "warning": (
                "The reported metrics apply to article-level out-of-fold "
                "evaluation only. They do not validate sentence highlighting, "
                "explanations, rewriting, or uncertainty display rules."
            ),
        },
        "artifact_compatibility": runtime.metadata,
    }
