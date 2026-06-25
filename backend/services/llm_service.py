import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MISSING_KEY_MESSAGE = (
    "ایل ایل ایم کی کلید دستیاب نہیں ہے۔ براہِ کرم بیک اینڈ کی .env فائل چیک کریں۔"
)
GENERIC_ERROR_MESSAGE = (
    "معذرت، اس وقت وضاحت تیار نہیں ہو سکی۔ کچھ دیر بعد دوبارہ کوشش کریں۔"
)


def get_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def _format_features(features: List[Dict[str, Any]]) -> str:
    words = []
    for item in features[:5]:
        feature = str(item.get("feature", "")).strip()
        if feature:
            words.append(feature)
    return "، ".join(words) or "کوئی نمایاں لفظ دستیاب نہیں"


def explain_bias(sentence_data):
    sentence = str(sentence_data.get("sentence", "")).strip()[:4_000]
    evidence_label = str(sentence_data.get("evidence_label", "uncertain"))
    top_features = sentence_data.get("top_features", [])
    semantic = sentence_data.get("semantic_signal", {})
    semantic_direction = str(semantic.get("direction", "unknown"))

    if not sentence:
        return "وضاحت کے لیے کوئی جملہ فراہم نہیں کیا گیا۔"

    client = get_client()
    if not client:
        return MISSING_KEY_MESSAGE

    if evidence_label == "bias_evidence":
        evidence_summary = "اندازِ بیان میں جانب داری کے آثار نمایاں ہیں"
    elif evidence_label == "neutral_evidence":
        evidence_summary = "اندازِ بیان نسبتاً متوازن اور غیر جانب دار ہے"
    else:
        evidence_summary = "اندازِ بیان کے بارے میں واضح نتیجہ اخذ نہیں ہوتا"

    direction_summary = {
        "biased": "زبان جانب داری کی طرف مائل محسوس ہوتی ہے",
        "unbiased": "زبان نسبتاً متوازن محسوس ہوتی ہے",
    }.get(semantic_direction, "مجموعی انداز واضح نہیں")

    prompt = f"""
آپ ایک تجربہ کار اردو نیوز ایڈیٹر ہیں۔ درج ذیل جملے کے اندازِ بیان کی وضاحت
فطری، رواں اور عام فہم اردو میں کریں۔

جملہ:
"{sentence}"

پس منظر کے لیے اندرونی خلاصہ:
- {evidence_summary}
- قابلِ توجہ الفاظ یا تراکیب: {_format_features(top_features)}
- مجموعی لسانی رخ: {direction_summary}

جواب کے اصول:
1. صرف ایک یا دو مختصر، فطری اردو جملے لکھیں۔
2. واضح کریں کہ زبان جذباتی، مبالغہ آمیز، قطعی، الزامیہ، یک طرفہ یا متوازن کیوں محسوس ہوتی ہے۔
3. صرف اصل جملے کے الفاظ اور اندازِ بیان پر بات کریں؛ بیرونی معلومات یا سیاسی مفروضے شامل نہ کریں۔
4. کسی عدد، فیصد، اسکور، امکان یا اعتماد کی شرح کا ذکر نہ کریں۔
5. مشین لرننگ، مصنوعی ذہانت، ماڈل، فیچر، سگنل، الگورتھم، درجہ بندی،
   احتمال، SHAP، TF-IDF یا LaBSE جیسی تکنیکی اصطلاحات استعمال نہ کریں۔
6. اگر زبان کے شواہد واضح نہ ہوں تو سادہ انداز میں کہیں کہ جملہ اکیلا کسی
   حتمی رائے کے لیے کافی نہیں۔
7. جواب میں سرخی، فہرست، انگریزی الفاظ یا تکنیکی وضاحت شامل نہ کریں۔

صرف حتمی اردو وضاحت لکھیں۔
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write natural Urdu editorial commentary grounded only in "
                        "the supplied sentence. Never expose scores, numerical "
                        "values, internal evidence labels, or machine-learning "
                        "terminology."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=180,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return content.strip() if content else GENERIC_ERROR_MESSAGE
    except Exception as exc:
        print(f"LLM explanation failed: {exc.__class__.__name__}")
        return GENERIC_ERROR_MESSAGE


def rewrite_unbiased(sentence):
    sentence = str(sentence).strip()[:4_000]
    if not sentence:
        return "دوبارہ لکھنے کے لیے کوئی جملہ فراہم نہیں کیا گیا۔"

    client = get_client()
    if not client:
        return MISSING_KEY_MESSAGE

    prompt = f"""
ذیل کے اردو جملے کو غیر جانب دار صحافتی انداز میں دوبارہ لکھیں:

"{sentence}"

قواعد:
1. اصل قابلِ مشاہدہ معلومات اور مرکزی مفہوم برقرار رکھیں۔
2. جذباتی، قطعی، توہین آمیز یا مبالغہ آمیز زبان نرم یا غیر جانب دار کریں۔
3. کوئی نیا دعویٰ، وجہ، عدد، نام یا سیاق شامل نہ کریں۔
4. اگر جملے میں دعویٰ کسی شخص سے منسوب ہے تو نسبت برقرار رکھیں۔
5. صرف دوبارہ لکھا ہوا اردو جملہ دیں، اضافی وضاحت نہیں۔
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite Urdu news language neutrally without inventing "
                        "or removing factual claims."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=180,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return content.strip() if content else GENERIC_ERROR_MESSAGE
    except Exception as exc:
        print(f"LLM rewrite failed: {exc.__class__.__name__}")
        return GENERIC_ERROR_MESSAGE
