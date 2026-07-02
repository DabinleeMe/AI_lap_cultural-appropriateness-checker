"""
01_Global_Ad_Prompt_Eval.py  —  Task 1, Step 3 (Evaluation)
============================================================
Runs the 3 prompt versions (A / B / C) from prompts.py against the Global
Advertising cultural-risk dataset, then scores each version on 3 tasks
(risk, topic, intent) with Accuracy, Macro-F1, and JSON parse-failure rate.
 
Dataset (loaded from CSV, per the handbook):
    Task1_Global_Advertising_Dataset.csv
    - text            : the marketing copy the model classifies
    - target_culture  : the culture to judge against
    - gt_risk         : ground-truth risk  (Expected / Normal / Taboo)
    - gt_topic        : ground-truth topic (messy labels -> normalised below)
    - gt_intent       : ground-truth intent (Intentional / Unintentional / Unclear)
The model NEVER sees the gt_* columns; they are only used for scoring.
 
Setup:
    pip install ollama pandas
For Ollama (free, local):
    ollama pull llama3.2:3b
    ollama serve
"""
 
import json
import os
#import re
import time
from collections import Counter
 
import pandas as pd
 
from prompts import build_prompt, RISK_LABELS, TOPIC_LABELS, INTENT_LABELS
 
# ── CONFIG — change these ────────────────────────────────────────────────────
 
DATA_FILE  = "Task1_Global_Advertising_Dataset.csv"  # CSV with text + gt_* labels
VERSIONS   = ["A", "B", "C"]          # which prompt versions to run/compare
 
MODEL      = "llama3.2:3b"            # local Ollama model name
 
# Which versions use Ollama's JSON mode (forces the reply to be a valid JSON object).
# A and B leave JSON formatting up to the 3B model, which it does badly, so we force it.
# C is intentionally left OFF: it must print free-text "Step-by-step:" reasoning before
# "FINAL: {json}", and JSON mode would suppress that reasoning entirely.
JSON_MODE_VERSIONS = ["A", "B"]
 
PRED_OUT   = "prompt_eval_predictions.csv"   # per-row predictions (all versions)
SUMMARY_OUT= "prompt_eval_summary.csv"       # version x task metric table
 
# ── TOPIC NORMALISATION ──────────────────────────────────────────────────────
# The dataset's gt_topic uses more detailed labels than the 6 clean categories
# the model predicts. We map each raw label to one of the 6 so scoring is fair.
# "Culture-aware (...)" means the reference was handled WELL (not a violation),
# so its violated-topic is "None". Compound labels take their primary token.
 
TOPIC_MAP = {
    "Religion":                          "Religion",
    "Religion/History":                  "Religion",
    "Gender":                            "Gender",
    "Gender identity/Politics":          "Gender",
    "Race/Ethnicity":                    "Race/Ethnicity",
    "Race/History":                      "Race/Ethnicity",
    "Language/Wordplay":                 "Language/Wordplay",
    "History/Politics":                  "History/Politics",
    "Culture-aware (Festival)":          "None",
    "Culture-aware (Religion)":          "None",
    "Culture-aware (Festival/Religion)": "None",
    "Intent/Context (Tone)":             "None",
    "None":                              "None",
}
 
 
def normalise_gt_topic(raw):
    """Map a raw gt_topic label to one of the 6 clean TOPIC_LABELS.
 
    Empty/NaN gt_topic is expected: it marks the 'Expected' safe-copy rows that
    engage no cultural topic, so the correct gold label is 'None'.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "None"
    key = str(raw).strip()
    if key in ("", "nan"):
        return "None"
    if key in TOPIC_MAP:
        return TOPIC_MAP[key]
    print(f"  ⚠ unmapped gt_topic '{raw}' -> treated as 'None' (add it to TOPIC_MAP)")
    return "None"
 
 
# ── LLM CALLER (local Ollama, same style as 01_Prompting.py) ─────────────────
 
def call_ollama(system, user, force_json=False):
    """Call local Llama via Ollama. `system` may be None (Version A).
 
    If force_json=True, we pass format="json", which makes Ollama constrain the
    model's output to a syntactically valid JSON object (no missing braces, no
    unquoted values). This is what rescues A/B on a small model.
    """
    try:
        import ollama
        messages = []
        if system:                       # Version A has no system message
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "options": {"temperature": 0.0},
        }
        if force_json:
            kwargs["format"] = "json"     # ← forces valid JSON output
        resp = ollama.chat(**kwargs)
        return resp.message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"
 
 
# ── OUTPUT PARSING & SCORING HELPERS ─────────────────────────────────────────
 
def parse_json_answer(raw):
    """Pull the {risk, topic, intent, reason} object out of the model's reply.
 
    Handles: plain JSON, ```json fences, Version C's 'FINAL: {json}', and extra
    chatter around the JSON. Returns a dict, or None if nothing parseable.
    """
    if not raw:
        return None
    text = str(raw)
    # Version C prints reasoning then 'FINAL: {json}' — keep only the last one.
    if "FINAL:" in text:
        text = text.rsplit("FINAL:", 1)[1]
    # Drop code fences the model may add.
    text = text.replace("```json", "").replace("```", "")
    # Take the outermost {...} block and try to parse it.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None
 
 
def norm_pred(value, canonical_labels):
    """Match a predicted label to canonical casing; else flag it."""
    if value is None:
        return "PARSE_FAIL"
    v = str(value).strip().lower()
    for lab in canonical_labels:
        if v == lab.lower():
            return lab
    return "INVALID"
 
 
def macro_f1(y_true, y_pred, labels):
    """Unweighted mean of per-class F1 over `labels` (no sklearn needed)."""
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0
 
 
def accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0
 
 
def classify_status(raw, parsed, preds):
    """Explain what happened to ONE response, for error analysis:
      ollama_error   -> the model call itself failed (Ollama not running, etc.)
      json_parse_fail-> model replied, but no valid JSON could be extracted
      bad_label      -> valid JSON, but a label wasn't one of the allowed values
      ok             -> clean, usable answer
    """
    if isinstance(raw, str) and raw.startswith("ERROR:"):
        return "ollama_error"
    if parsed is None:
        return "json_parse_fail"
    if "INVALID" in preds:
        return "bad_label"
    return "ok"
 
 
# ── LOAD DATA ────────────────────────────────────────────────────────────────
 
print("=" * 60)
print("  Task 1 — Prompt A/B/C Evaluation (Global Advertising)")
print(f"  Model: {MODEL}  (local via Ollama)")
print("=" * 60)
 
def _find_data_file(name):
    """Find the CSV next to this script, or in a Data/ subfolder, or in the
    current working directory — so it runs no matter where you launch it from."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, name),
                 os.path.join(here, "Data", name),
                 name):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find '{name}'. I looked next to the script and in a Data/ "
        f"subfolder. Put the CSV in one of those places, or set DATA_FILE at the "
        f"top to a full path like '/Users/tabami/Desktop/AI_Lap_Project/{name}'."
    )
 
data_path = _find_data_file(DATA_FILE)
df = pd.read_csv(data_path)
df["topic_gold"] = df["gt_topic"].apply(normalise_gt_topic)
print(f"\n✓ Loaded '{data_path}'  ({len(df)} rows)\n")
 
# ── RUN EACH VERSION ─────────────────────────────────────────────────────────
 
all_rows   = []   # per-row predictions across all versions (for PRED_OUT)
summary    = []   # version x task metrics (for SUMMARY_OUT)
json_fail  = {}   # version -> parse-failure count
 
for version in VERSIONS:
    print(f"{'─'*60}\n  VERSION {version}\n{'─'*60}")
 
    rows, fails = [], 0
    for i, r in df.iterrows():
        system, user = build_prompt(version, text=str(r["text"]),
                                    target_culture=str(r["target_culture"]))
        raw    = call_ollama(system, user, force_json=(version in JSON_MODE_VERSIONS))
        parsed = parse_json_answer(raw)
        if parsed is None:
            fails += 1
 
        pred_risk   = norm_pred(parsed.get("risk")   if parsed else None, RISK_LABELS)
        pred_topic  = norm_pred(parsed.get("topic")  if parsed else None, TOPIC_LABELS)
        pred_intent = norm_pred(parsed.get("intent") if parsed else None, INTENT_LABELS)
        status = classify_status(raw, parsed, [pred_risk, pred_topic, pred_intent])
 
        rows.append({
            "id": r["id"], "version": version, "text": r["text"],
            "gt_risk": r["gt_risk"],   "pred_risk": pred_risk,
            "gt_topic": r["topic_gold"], "pred_topic": pred_topic,
            "gt_intent": r["gt_intent"], "pred_intent": pred_intent,
            "json_ok": parsed is not None,
            "status": status,
            "raw": raw,               # the model's exact reply (for error analysis)
        })
 
        mark = "✓" if pred_risk == r["gt_risk"] else "✗"
        snippet = str(r["text"])[:44]
        print(f"  [{version}][{i+1:02d}] risk {mark} "
              f"({str(r['gt_risk']):8}->{pred_risk:10}) | {snippet}")
        time.sleep(0.05)
 
    all_rows.extend(rows)
    json_fail[version] = fails
 
    # Score the 3 tasks for this version.
    for task, gt_key, pr_key, labels in [
        ("risk",   "gt_risk",   "pred_risk",   RISK_LABELS),
        ("topic",  "gt_topic",  "pred_topic",  TOPIC_LABELS),
        ("intent", "gt_intent", "pred_intent", INTENT_LABELS),
    ]:
        y_true = [x[gt_key] for x in rows]
        y_pred = [x[pr_key] for x in rows]
        summary.append({
            "version": version, "task": task,
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "macro_f1": round(macro_f1(y_true, y_pred, labels), 4),
            "json_fail_rate": round(fails / len(rows), 4),
        })
 
    fr = fails / len(rows) * 100
    print(f"\n  Version {version}: JSON parse-failures = {fails}/{len(rows)} ({fr:.1f}%)\n")
 
# ── COMPARISON TABLES ────────────────────────────────────────────────────────
 
summary_df = pd.DataFrame(summary)
 
print("=" * 60)
print("  COMPARISON — Accuracy & Macro-F1 by version")
print("=" * 60)
for task in ["risk", "topic", "intent"]:
    print(f"\n  TASK: {task.upper()}")
    print(f"    {'Ver':<5}{'Accuracy':>12}{'Macro-F1':>12}")
    for v in VERSIONS:
        row = summary_df[(summary_df.version == v) & (summary_df.task == task)].iloc[0]
        print(f"    {v:<5}{row.accuracy*100:>10.1f}%{row.macro_f1:>12.3f}")
 
print("\n  JSON parse-failure rate (stability):")
for v in VERSIONS:
    print(f"    {v}: {json_fail[v]/len(df)*100:5.1f}%   ({json_fail[v]}/{len(df)})")
 
# ── WHY DID ANSWERS FAIL? (error analysis) ───────────────────────────────────
# Break each version down by status, and show a couple of real failed replies so
# you can SEE what A/B actually produced instead of clean JSON.
 
print("\n" + "=" * 60)
print("  FAILURE BREAKDOWN — what happened to each version's answers")
print("=" * 60)
for v in VERSIONS:
    sub = [x for x in all_rows if x["version"] == v]
    counts = Counter(x["status"] for x in sub)
    print(f"\n  VERSION {v}: " + ", ".join(f"{k}={n}" for k, n in counts.most_common()))
    samples = [x for x in sub if x["status"] != "ok"][:2]
    for s in samples:
        preview = " ".join(str(s["raw"]).split())[:200]   # collapse whitespace
        print(f"    e.g. id{s['id']} [{s['status']}]: {preview}")
 
# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
 
pd.DataFrame(all_rows).to_csv(PRED_OUT, index=False)
summary_df.to_csv(SUMMARY_OUT, index=False)
 
print("\n" + "=" * 60)
print(f"  Saved per-row predictions -> {PRED_OUT}")
print(f"  Saved metric summary      -> {SUMMARY_OUT}")
print("=" * 60)