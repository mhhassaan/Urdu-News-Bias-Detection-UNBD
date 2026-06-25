"use client";

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Lang = "en" | "ur";
type EvidenceLabel = "bias_evidence" | "neutral_evidence" | "uncertain";

type FeatureContribution = {
  feature: string;
  contribution: number;
  direction: "biased" | "unbiased";
};

type ExplanationComponent = {
  signed_contribution: number;
  magnitude: number;
  direction: "biased" | "unbiased";
  top_features?: FeatureContribution[];
  note?: string;
};

type HybridExplanation = {
  output_space: string;
  lexical: ExplanationComponent;
  semantic: ExplanationComponent & { dimensions?: number };
  total_feature_contribution: number;
};

type SentenceScore = {
  index: number;
  sentence: string;
  context: string;
  context_span: { start: number; end: number };
  sentence_probability: number;
  context_probability: number;
  score: number;
  prediction: "biased" | "unbiased";
  confidence: number;
  evidence_label: EvidenceLabel;
  top_features: FeatureContribution[];
  semantic_signal: ExplanationComponent;
  scope_note: string;
};

type PredictResult = {
  prediction: "biased" | "unbiased";
  decision_status: "confident" | "uncertain";
  confidence: number;
  biased_probability: number;
  uncertainty: {
    status: "stable" | "review" | "uncertain";
    severity: "low" | "medium" | "high";
    review_recommended: boolean;
    reasons: string[];
    probability_is_calibrated: boolean;
    note: string;
  };
  article_explanation: HybridExplanation;
  sentence_scores: SentenceScore[];
  input_profile: {
    word_count: number;
    sentence_count: number;
    lexical_coverage: { known: number; total: number; ratio: number };
    training_reference: {
      samples: number;
      word_count_p95: number;
      word_count_max: number;
    };
  };
  evaluation_scope: {
    reported_cross_validation_accuracy: number;
    validated_unit: "article";
    sentence_level_metrics_available: false;
    warning: string;
  };
  text_preview?: string;
};

type LlmEntry = {
  explanation?: string;
  rewritten?: string;
  loadingAction?: "explain" | "rewrite";
};

const copy = {
  en: {
    textLabel: "Paste Urdu article text",
    urlLabel: "Or analyze a public news URL",
    placeholderText: "Paste Urdu news text or an article here...",
    placeholderUrl: "https://example.com/urdu-news-article",
    toggle: "اردو",
    explain: "GROUNDED EXPLANATION",
    explaining: "EXPLAINING…",
    explanationTitle: "EXPLANATION",
    rewrite: "NEUTRAL REWRITE",
    rewriting: "REWRITING…",
    rewriteTitle: "NEUTRAL DRAFT",
  },
  ur: {
    textLabel: "اردو خبر یا مضمون درج کریں",
    urlLabel: "یا عوامی نیوز لنک کا تجزیہ کریں",
    placeholderText: "اردو خبر یا مضمون یہاں درج کریں۔۔۔",
    placeholderUrl: "https://example.com/urdu-news-article",
    toggle: "English",
    explain: "وضاحت حاصل کریں",
    explaining: "وضاحت تیار ہو رہی ہے۔۔۔",
    explanationTitle: "وضاحت",
    rewrite: "غیر جانب دار تحریر",
    rewriting: "دوبارہ لکھا جا رہا ہے۔۔۔",
    rewriteTitle: "غیر جانب دار متن",
  },
} satisfies Record<Lang, Record<string, string>>;

const evidenceCopy: Record<
  EvidenceLabel,
  { label: string; className: string; railClassName: string }
> = {
  bias_evidence: {
    label: "BIAS EVIDENCE",
    className: "bg-red-50 text-red-800",
    railClassName: "border-red-700",
  },
  neutral_evidence: {
    label: "NEUTRAL EVIDENCE",
    className: "bg-emerald-50 text-emerald-900",
    railClassName: "border-emerald-700",
  },
  uncertain: {
    label: "MIXED / WEAK EVIDENCE",
    className: "bg-amber-50 text-amber-900",
    railClassName: "border-amber-600",
  },
};

async function readApiError(response: Response, fallback: string) {
  try {
    const data = (await response.json()) as { detail?: string };
    return data.detail ?? fallback;
  } catch {
    return fallback;
  }
}

function getErrorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function probabilityTone(probability: number) {
  if (probability >= 0.6) return "text-red-700";
  if (probability <= 0.4) return "text-emerald-800";
  return "text-amber-700";
}

export default function Home() {
  const [lang, setLang] = useState<Lang>("en");
  const [inputText, setInputText] = useState("");
  const [inputUrl, setInputUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");
  const [llmData, setLlmData] = useState<Record<number, LlmEntry>>({});
  const resultRef = useRef<HTMLElement>(null);

  const t = copy[lang];
  const isUrdu = lang === "ur";

  useEffect(() => {
    document.documentElement.lang = "en";
    document.documentElement.dir = "ltr";
  }, []);

  useEffect(() => {
    if (result) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [result]);

  const handlePredict = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    const payload = inputUrl.trim()
      ? { url: inputUrl.trim() }
      : { text: inputText.trim() };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await readApiError(response, "The analysis request failed."));
      }
      setResult((await response.json()) as PredictResult);
      setLlmData({});
    } catch (requestError) {
      setError(getErrorMessage(requestError, "The analysis request failed."));
    } finally {
      setLoading(false);
    }
  };

  const handleNewAnalysis = () => {
    setResult(null);
    setError("");
    setInputText("");
    setInputUrl("");
    setLlmData({});
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleLlmAction = async (
    action: "explain" | "rewrite",
    item: SentenceScore,
  ) => {
    setLlmData((current) => ({
      ...current,
      [item.index]: { ...current[item.index], loadingAction: action },
    }));

    const body =
      action === "explain"
        ? JSON.stringify({ data: item })
        : JSON.stringify({ sentence: item.sentence });

    try {
      const response = await fetch(`${API_URL}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!response.ok) {
        throw new Error(await readApiError(response, "The LLM request failed."));
      }

      const data = (await response.json()) as {
        explanation?: string;
        rewritten?: string;
      };
      setLlmData((current) => ({
        ...current,
        [item.index]: {
          ...current[item.index],
          explanation:
            action === "explain"
              ? (data.explanation ?? "No explanation returned.")
              : current[item.index]?.explanation,
          rewritten:
            action === "rewrite"
              ? (data.rewritten ?? "No rewrite returned.")
              : current[item.index]?.rewritten,
          loadingAction: undefined,
        },
      }));
    } catch (requestError) {
      const message = getErrorMessage(requestError, "The LLM request failed.");
      setLlmData((current) => ({
        ...current,
        [item.index]: {
          ...current[item.index],
          [action === "explain" ? "explanation" : "rewritten"]: message,
          loadingAction: undefined,
        },
      }));
    }
  };

  const statusLabel = result
    ? result.decision_status === "uncertain"
      ? `INCONCLUSIVE — LEANS ${result.prediction.toUpperCase()}`
      : result.prediction.toUpperCase()
    : "";

  return (
    <div className="min-h-screen bg-[#f4f0e7] text-[#171713]">
      <div className="border-b border-black bg-black px-5 py-3 text-[#f4f0e7]">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="kicker text-[10px]">UNBD</div>
          <button
            type="button"
            onClick={() => setLang((current) => (current === "en" ? "ur" : "en"))}
            className={`kicker border border-stone-500 px-3 py-1 text-[10px] transition hover:border-white ${
              isUrdu ? "font-urdu text-sm" : ""
            }`}
          >
            {t.toggle}
          </button>
        </div>
      </div>

      <header className="relative overflow-hidden border-b-4 border-black px-5 py-14 md:py-20">
        <div className="paper-grid absolute inset-0 opacity-40" />
        <div className="relative mx-auto grid max-w-6xl gap-8 md:grid-cols-[1fr_18rem] md:items-end">
          <div>
            <h1 className="max-w-4xl font-playfair text-5xl font-black leading-[0.92] tracking-[-0.055em] md:text-8xl">
              URDU NEWS BIAS DETECTOR
            </h1>
            <p className="mt-6 max-w-3xl font-lora text-lg italic text-stone-700 md:text-2xl">
              Article-level classification with contextual evidence and transparent limits.
            </p>
          </div>
          <aside className="border-l-4 border-[#db2b19] pl-5" dir="ltr">
            <div className="kicker text-[#a51e12]">INTERPRETATION NOTE</div>
            <div className="mt-2 font-playfair text-4xl font-black">REVIEW, DON’T ASSUME</div>
            <p className="mt-2 text-sm leading-relaxed text-stone-700">
              Use the result as supporting evidence. Uncertain or conflicting passages should
              always receive human review.
            </p>
          </aside>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-12 md:py-16">
        {error ? (
          <div className="mb-10 border-2 border-red-800 bg-red-50 p-5" role="alert">
            <div className="kicker text-red-800">REQUEST ERROR</div>
            <p className="mt-2 text-lg font-bold">{error}</p>
          </div>
        ) : null}

        {!result ? (
          <section className="grid gap-10 md:grid-cols-[1fr_18rem]">
            <form onSubmit={handlePredict} className="border-2 border-black bg-[#fffdf7] p-5 md:p-8">
              <label
                className={`mb-3 block font-bold ${isUrdu ? "font-urdu text-xl" : "kicker"}`}
                htmlFor="article-text"
                dir={isUrdu ? "rtl" : "ltr"}
              >
                {t.textLabel}
              </label>
              <textarea
                id="article-text"
                rows={10}
                dir={isUrdu ? "rtl" : "ltr"}
                className={`w-full resize-y border-2 border-black bg-transparent p-5 text-xl leading-loose outline-none transition focus:bg-amber-50 ${
                  isUrdu ? "font-urdu text-right" : "font-lora text-left"
                }`}
                placeholder={t.placeholderText}
                value={inputText}
                onChange={(event) => {
                  setInputText(event.target.value);
                  if (event.target.value) setInputUrl("");
                }}
              />

              <div className="my-6 flex items-center gap-4" dir="ltr">
                <div className="h-px flex-1 bg-black" />
                <span className="kicker">OR</span>
                <div className="h-px flex-1 bg-black" />
              </div>

              <label
                className={`mb-3 block font-bold ${isUrdu ? "font-urdu text-xl" : "kicker"}`}
                htmlFor="article-url"
                dir={isUrdu ? "rtl" : "ltr"}
              >
                {t.urlLabel}
              </label>
              <input
                id="article-url"
                type="url"
                dir="ltr"
                className="w-full border-2 border-black bg-transparent p-4 font-mono text-sm outline-none transition focus:bg-amber-50"
                placeholder={t.placeholderUrl}
                value={inputUrl}
                onChange={(event) => {
                  setInputUrl(event.target.value);
                  if (event.target.value) setInputText("");
                }}
              />

              <button
                type="submit"
                disabled={loading || (!inputText.trim() && !inputUrl.trim())}
                className="kicker mt-7 w-full border-2 border-black bg-black px-5 py-4 text-lg font-black text-white transition hover:bg-[#db2b19] disabled:cursor-not-allowed disabled:bg-stone-400"
              >
                {loading ? "ANALYZING…" : "RUN ARTICLE ANALYSIS"}
              </button>
            </form>

            <aside className="space-y-8" dir="ltr">
              <div>
                <div className="kicker border-b border-black pb-2">HOW TO USE</div>
                <p className="mt-4 text-sm leading-relaxed text-stone-700">
                  Paste an Urdu article or provide a public news link. The result will show an
                  overall assessment and the passages that deserve closer review.
                </p>
              </div>
              <div className="border-t-4 border-black pt-4">
                <div className="kicker">READ RESULTS CAREFULLY</div>
                <p className="mt-3 text-sm leading-relaxed text-stone-700">
                  The percentage is an analysis score, not guaranteed certainty. Yellow results
                  and review warnings should be treated as inconclusive.
                </p>
              </div>
            </aside>
          </section>
        ) : (
          <section ref={resultRef} className="scroll-mt-6" dir="ltr">
            <div className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b-4 border-black pb-4">
              <div>
                <div className="kicker text-stone-500">ARTICLE ANALYSIS</div>
                <h2 className="font-playfair text-4xl font-black md:text-5xl">Evidence dossier</h2>
              </div>
              <button
                type="button"
                onClick={handleNewAnalysis}
                className="kicker border-2 border-black px-5 py-3 transition hover:bg-black hover:text-white"
              >
                NEW ANALYSIS
              </button>
            </div>

            <div>
              <article className="border-2 border-black bg-[#fffdf7] p-6 md:p-8">
                <div className="kicker text-stone-500">ARTICLE-LEVEL DECISION</div>
                <div
                  className={`mt-3 font-playfair text-4xl font-black leading-none md:text-6xl ${probabilityTone(
                    result.biased_probability,
                  )}`}
                >
                  {statusLabel}
                </div>
                <p className="mt-4 max-w-2xl text-sm leading-relaxed text-stone-700">
                  This label is produced from the complete article input. It is not assembled
                  from sentence votes.
                </p>

                <div className="mt-8">
                  <div className="flex justify-between font-mono text-[10px] font-bold">
                    <span>UNBIASED</span>
                    <span>INDETERMINATE 40–60%</span>
                    <span>BIASED</span>
                  </div>
                  <div className="relative mt-2 h-8 border-2 border-black bg-gradient-to-r from-emerald-200 via-amber-200 to-red-200">
                    <div className="absolute bottom-0 left-[40%] top-0 border-l border-black/40" />
                    <div className="absolute bottom-0 left-[60%] top-0 border-l border-black/40" />
                    <div
                      className="absolute -top-2 h-11 w-1 bg-black"
                      style={{ left: `calc(${result.biased_probability * 100}% - 2px)` }}
                    />
                  </div>
                  <div className="mt-3 flex items-end justify-between">
                    <div>
                      <div className="kicker text-stone-500">BIASED-CLASS PROBABILITY</div>
                      <div className="font-playfair text-4xl font-black">
                        {percent(result.biased_probability)}
                      </div>
                    </div>
                    <div className="max-w-xs text-right text-xs leading-relaxed text-stone-600">
                      Not separately calibrated; do not read this as real-world certainty.
                    </div>
                  </div>
                </div>
              </article>
            </div>

            <div className="mt-6 grid gap-px border-2 border-black bg-black sm:grid-cols-4">
              {[
                ["WORDS", result.input_profile.word_count.toLocaleString()],
                ["SENTENCES", result.input_profile.sentence_count.toLocaleString()],
                ["ASSESSMENT", statusLabel],
                ["EVIDENCE ITEMS", result.sentence_scores.length.toLocaleString()],
              ].map(([label, value]) => (
                <div key={label} className="bg-[#fffdf7] p-5">
                  <div className="kicker text-[10px] text-stone-500">{label}</div>
                  <div className="mt-2 font-playfair text-2xl font-black">{value}</div>
                </div>
              ))}
            </div>

            <section className="mt-14">
              <div className="mb-6 grid gap-5 border-b-4 border-black pb-5 md:grid-cols-[1fr_22rem]">
                <div>
                  <div className="kicker text-[#a51e12]">ARTICLE PASSAGES</div>
                  <h3 className="mt-2 font-playfair text-4xl font-black">Review the article</h3>
                </div>
                <p className="text-sm leading-relaxed text-stone-700">
                  Read the highlighted passages and request a plain-language explanation where
                  needed.
                </p>
              </div>

              <div className="space-y-4">
                {result.sentence_scores.map((item) => {
                  const visual = evidenceCopy[item.evidence_label];
                  const llm = llmData[item.index];
                  return (
                    <article
                      key={`${item.index}-${item.sentence}`}
                      className={`border-2 border-black border-r-8 bg-[#fffdf7] ${visual.railClassName}`}
                    >
                      <div className="grid md:grid-cols-[8rem_1fr]">
                        <div className={`border-b border-black p-4 md:border-b-0 md:border-r ${visual.className}`}>
                          <div className="kicker text-[9px]">{visual.label}</div>
                          <div className="mt-4 border-t border-black/20 pt-3 font-mono text-[9px]">
                            CONTEXTUAL REVIEW
                          </div>
                        </div>

                        <div className="p-5 md:p-6">
                          <p className="font-urdu text-xl leading-[2.2] md:text-2xl" dir="rtl">
                            {item.sentence}
                          </p>

                          <details className="mt-5 border-t border-stone-300 pt-3">
                            <summary className="kicker cursor-pointer text-[9px]">
                              SHOW CONTEXT WINDOW · SENTENCES {item.context_span.start + 1}–
                              {item.context_span.end + 1}
                            </summary>
                            <p className="mt-3 bg-stone-100 p-4 font-urdu text-base leading-loose" dir="rtl">
                              {item.context}
                            </p>
                          </details>

                          <div className="mt-5 flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={() => handleLlmAction("explain", item)}
                              disabled={Boolean(llm?.loadingAction)}
                              className="kicker border border-black bg-black px-3 py-2 text-[9px] text-white hover:bg-[#db2b19] disabled:opacity-40"
                            >
                              {llm?.loadingAction === "explain" ? t.explaining : t.explain}
                            </button>
                            {item.evidence_label === "bias_evidence" ? (
                              <button
                                type="button"
                                onClick={() => handleLlmAction("rewrite", item)}
                                disabled={Boolean(llm?.loadingAction)}
                                className="kicker border border-black px-3 py-2 text-[9px] hover:bg-emerald-800 hover:text-white disabled:opacity-40"
                              >
                                {llm?.loadingAction === "rewrite" ? t.rewriting : t.rewrite}
                              </button>
                            ) : null}
                          </div>

                          {llm?.explanation ? (
                            <div className="mt-4 border-l-4 border-red-700 bg-red-50 p-4">
                              <div
                                className={`text-red-800 ${isUrdu ? "font-urdu text-base" : "kicker text-[9px]"}`}
                                dir={isUrdu ? "rtl" : "ltr"}
                              >
                                {t.explanationTitle}
                              </div>
                              <p className="mt-2 font-urdu text-lg leading-loose" dir="rtl">
                                {llm.explanation}
                              </p>
                            </div>
                          ) : null}
                          {llm?.rewritten ? (
                            <div className="mt-4 border-l-4 border-emerald-700 bg-emerald-50 p-4">
                              <div
                                className={`text-emerald-900 ${isUrdu ? "font-urdu text-base" : "kicker text-[9px]"}`}
                                dir={isUrdu ? "rtl" : "ltr"}
                              >
                                {t.rewriteTitle}
                              </div>
                              <p className="mt-2 font-urdu text-lg leading-loose" dir="rtl">
                                {llm.rewritten}
                              </p>
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>

            <aside className="mt-12 border-4 border-black bg-amber-100 p-6 md:p-8">
              <div className="kicker">RESPONSIBLE INTERPRETATION</div>
              <p className="mt-3 max-w-4xl font-playfair text-2xl font-bold leading-snug">
                This analysis is decision support, not a substitute for editorial judgment.
              </p>
              <p className="mt-4 max-w-4xl text-sm leading-relaxed text-stone-700">
                Review the original article, source attribution, surrounding context, and
                uncertain passages before making a final assessment.
              </p>
            </aside>
          </section>
        )}
      </main>

      <footer className="mt-16 border-t-4 border-black bg-black px-5 py-10 text-stone-300" dir="ltr">
        <div className="mx-auto grid max-w-6xl gap-8 md:grid-cols-3">
          <div>
            <div className="kicker text-white">METHOD</div>
            <p className="mt-3 text-sm">Article-level Urdu news analysis.</p>
          </div>
          <div>
            <div className="kicker text-white">EVIDENCE</div>
            <p className="mt-3 text-sm">Contextual passages and language-based explanations.</p>
          </div>
          <div>
            <div className="kicker text-white">LIMIT</div>
            <p className="mt-3 text-sm">Results should be confirmed through human editorial review.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
