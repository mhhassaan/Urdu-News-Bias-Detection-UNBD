import sys
import types
import unittest
from pathlib import Path

import numpy as np


# Policy tests should not load multi-gigabyte ML dependencies.
joblib_stub = types.ModuleType("joblib")
shap_stub = types.ModuleType("shap")
torch_stub = types.ModuleType("torch")
torch_stub.device = object
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModel = object
transformers_stub.AutoTokenizer = object
sys.modules.setdefault("joblib", joblib_stub)
sys.modules.setdefault("shap", shap_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("transformers", transformers_stub)

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from services.predictor import (  # noqa: E402
    _evidence_label,
    _uncertainty_assessment,
    context_window,
    decision_from_probability,
    split_sentences,
)


class PredictionPolicyTests(unittest.TestCase):
    def test_urdu_sentence_split_keeps_terminal_punctuation(self):
        text = "یہ پہلا جملہ ہے۔ کیا یہ دوسرا جملہ ہے؟ ہاں!"
        self.assertEqual(
            split_sentences(text),
            ["یہ پہلا جملہ ہے۔", "کیا یہ دوسرا جملہ ہے؟", "ہاں!"],
        )

    def test_context_window_uses_previous_current_and_next(self):
        context, span = context_window(["ایک", "دو", "تین", "چار"], 2)
        self.assertEqual(context, "دو تین چار")
        self.assertEqual(span, {"start": 1, "end": 3})

    def test_probability_band_is_explicitly_uncertain(self):
        decision = decision_from_probability(0.55)
        self.assertEqual(decision["label"], "biased")
        self.assertEqual(decision["status"], "uncertain")
        self.assertTrue(decision["is_uncertain"])

    def test_context_highlight_requires_agreement(self):
        self.assertEqual(_evidence_label(0.70, 0.75), "bias_evidence")
        self.assertEqual(_evidence_label(0.30, 0.25), "neutral_evidence")
        self.assertEqual(_evidence_label(0.45, 0.75), "uncertain")

    def test_conflicting_contexts_trigger_review(self):
        decision = decision_from_probability(0.70)
        assessment = _uncertainty_assessment(
            decision,
            word_count=60,
            lexical_coverage={"known": 8, "total": 10, "ratio": 0.8},
            context_probabilities=np.array([0.20, 0.75, 0.25, 0.80]),
        )
        self.assertEqual(assessment["status"], "review")
        self.assertTrue(assessment["review_recommended"])
        self.assertTrue(
            any("conflicting" in reason for reason in assessment["reasons"])
        )


if __name__ == "__main__":
    unittest.main()
