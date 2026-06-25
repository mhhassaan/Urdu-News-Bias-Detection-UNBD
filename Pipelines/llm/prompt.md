Act as an expert linguistic data analyst specialized in bias detection and content categorization in Urdu text.
Your task is to classify each Urdu sentence from a dataset into one of two categories:

* biased
* unbiased
Context: Each input row contains a single Urdu sentence. Your output will be stored in a CSV file with the following columns:

* text (original sentence)
* llm (your predicted label)
* label (ground truth for evaluation)
You must carefully analyze each sentence based on linguistic bias patterns and assign the correct label in the llm column.
GUIDELINES FOR BIAS DETECTION:
A sentence is labeled as BIASED if it contains any of the following:

1. Framing Bias — Subjective Intensifiers Words that express opinion, emotion, or exaggeration.
2. Framing Bias — One-Sided Terms Language that presents only one perspective on a controversial issue.
3. Epistemological Bias — Factive Verbs Verbs that assume truth of an unverified claim .
4. Epistemological Bias — Assertive Verbs Verbs that imply opinion, doubt, or attribution of belief.
5. Epistemological Bias — Entailments Words that embed judgment or accusation beyond literal meaning.
A sentence is UNBIASED if:

* It is neutral in tone
* It does not assume unverified truth
* It presents information without emotional or ideological framing
* It does not favor any side
INSTRUCTIONS:

* Analyze each sentence carefully
* Use reasoning internally but DO NOT display it
* Do NOT explain your answer
* Do NOT output anything except the final label
OUTPUT FORMAT (STRICT):
Return ONLY one word per input sentence: biased OR unbiased in the llm column and give the labelled .csv file

```
