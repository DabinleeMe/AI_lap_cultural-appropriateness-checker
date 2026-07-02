"""
Customer Sentiment Analysis using Local Llama (Ollama) or Gemini API
=====================================================================
Upload your labelled CSV, run prompts, get predictions + summary.
 
CSV format expected:
    text, label
    "I love this product!", positive
    "Terrible service.", negative
 
Setup:
    pip install ollama google-generativeai pandas
 
For Ollama (free, local):
    ollama pull llama3.2:3b
    ollama serve
"""
 
import pandas as pd
import time
import os
 
# ── CONFIG — change these ────────────────────────────────────────────────────
 
CSV_FILE   = "customer_reviews.csv"   # your CSV file path
TEXT_COL   = "text"                   # column name for review text
LABEL_COL  = "label"                  # column name for true label
 
BACKEND    = "ollama"                 # "ollama" or "gemini"
MODEL      = "llama3.2:3b"           # Ollama model name
GEMINI_KEY = "your-gemini-api-key"   # only needed if BACKEND = "gemini"
 
# ── SAMPLE DATA (used if CSV not found) ──────────────────────────────────────
 
SAMPLE_DATA = pd.DataFrame({
    "text": [
        "I absolutely love this product! Best purchase I've made.",
        "Terrible quality. Broke after one week. Very disappointed.",
        "It's okay. Nothing special but does the job.",
        "Amazing customer service! They resolved my issue instantly.",
        "Never buying from this store again. Complete waste of money.",
        "Decent product for the price. Arrived on time.",
        "The packaging was damaged but the item was fine.",
        "Outstanding! Exceeded all my expectations.",
        "Very slow delivery. Took three weeks instead of three days.",
        "Pretty good overall. Minor issues but nothing serious.",
    ],
    "label": [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "neutral",  "positive", "negative", "neutral",
    ]
})
 
# ── PROMPTS ───────────────────────────────────────────────────────────────────
 
SYSTEM_PROMPT = """You are a sentiment analysis assistant.
Classify customer reviews as exactly one of: positive, negative, or neutral.
Reply with ONLY the single word. No explanation."""
 
def make_sentiment_prompt(text):
    return f"Review: {text}\nSentiment:"
 
SUMMARY_SYSTEM = """You are a customer insights analyst.
Given a list of customer reviews, write a 3-sentence business summary:
1. Overall sentiment breakdown
2. Main customer complaints
3. Main customer praise points"""
 
def make_summary_prompt(reviews_text):
    return f"Customer reviews:\n{reviews_text}\n\nBusiness summary:"
 
# ── LLM CALLERS ───────────────────────────────────────────────────────────────
 
def call_ollama(system, user):
    """Call local Llama via Ollama."""
    try:
        import ollama
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            options={"temperature": 0.0}
        )
        return resp.message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"
 
 
def call_gemini(system, user):
    """Call Gemini via Google API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system
        )
        resp = model.generate_content(user)
        return resp.text.strip()
    except Exception as e:
        return f"ERROR: {e}"
 
 
def ask_llm(system, user):
    """Route to selected backend."""
    if BACKEND == "gemini":
        return call_gemini(system, user)
    else:
        return call_ollama(system, user)
 
 
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
 
print("=" * 55)
print("  Customer Sentiment Analysis with LLM Prompts")
print(f"  Backend: {BACKEND.upper()}  |  Model: {MODEL if BACKEND=='ollama' else 'gemini-1.5-flash'}")
print("=" * 55)
 
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    print(f"\n✓ Loaded '{CSV_FILE}'  ({len(df)} rows)")
else:
    df = SAMPLE_DATA.copy()
    print(f"\n⚠ '{CSV_FILE}' not found — using built-in sample data ({len(df)} rows)")
    print("  To use your own data: set CSV_FILE to your file path above.\n")
 
# ── TASK 1: SENTIMENT CLASSIFICATION ─────────────────────────────────────────
 
print(f"\n{'─'*55}")
print("  TASK 1: Sentiment Classification")
print(f"{'─'*55}")
 
predictions = []
for i, row in df.iterrows():
    text  = str(row[TEXT_COL])
    label = str(row[LABEL_COL]) if LABEL_COL in df.columns else "unknown"
 
    pred  = ask_llm(SYSTEM_PROMPT, make_sentiment_prompt(text))
 
    # Clean up output to single word
    pred_clean = pred.lower().split()[0].rstrip(".,!") if pred else "unknown"
    correct    = "✓" if pred_clean == label.lower() else "✗"
 
    predictions.append({
        "text":       text,
        "true_label": label,
        "predicted":  pred_clean,
        "correct":    correct,
    })
 
    print(f"  [{i+1:02d}] {correct} True: {label:<10} | Pred: {pred_clean:<10} | {text[:45]}…" if len(text)>45 else
          f"  [{i+1:02d}] {correct} True: {label:<10} | Pred: {pred_clean:<10} | {text}")
    time.sleep(0.1)   # small delay between calls
 
# ── ACCURACY ──────────────────────────────────────────────────────────────────
 
results_df = pd.DataFrame(predictions)
correct_count = sum(1 for r in predictions if r["correct"] == "✓")
accuracy      = correct_count / len(predictions) * 100
 
print(f"\n  Accuracy: {correct_count}/{len(predictions)} = {accuracy:.1f}%")
 
# Breakdown by label
print("\n  Per-class breakdown:")
for label in sorted(results_df["true_label"].unique()):
    sub      = results_df[results_df["true_label"] == label]
    correct  = sum(sub["correct"] == "✓")
    print(f"    {label:<12}: {correct}/{len(sub)} correct ({correct/len(sub)*100:.0f}%)")
 
# ── TASK 2: SUMMARY ───────────────────────────────────────────────────────────
 
print(f"\n{'─'*55}")
print("  TASK 2: Business Summary from All Reviews")
print(f"{'─'*55}")
 
all_reviews = "\n".join(
    f"- [{r['true_label']}] {r['text']}"
    for r in predictions
)
 
summary = ask_llm(SUMMARY_SYSTEM, make_summary_prompt(all_reviews))
print(f"\n{summary}\n")
 
# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
 
out_file = "sentiment_results.csv"
results_df.to_csv(out_file, index=False)
print(f"{'─'*55}")
print(f"  Results saved to: {out_file}")
print(f"  Accuracy: {accuracy:.1f}%")
print("=" * 55)
 