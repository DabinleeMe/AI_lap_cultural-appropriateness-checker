"""
analyze_results.py  —  Analyze the evaluation results yourself
================================================================
[What this script does]
Reads the per-row predictions file (prompt_eval_predictions.csv) and,
for each of the three prompt versions (A/B/C) on each task
(risk / topic / intent), computes and prints:
  - accuracy
  - macro-F1
  - per-class precision (P) / recall (R) / F1
  - a confusion matrix (actual vs predicted counts)

[How to run]  In your terminal:
    cd ~/Desktop/AI_Lap_Project
    source .venv/bin/activate      # if you use a venv
    python analyze_results.py

* You only need pandas. If it is missing:  python -m pip install pandas
================================================================
"""

import os
import pandas as pd


# ── 1. The file we want to analyze ──────────────────────────────
# This is the "per-row predictions" file the grading script produced.
PRED_FILE = "prompt_eval_predictions.csv"


# ── 2. The list of allowed labels for each task ─────────────────
# (Scoring only happens within these label sets.)
TASKS = [
    # (display name,     truth column, prediction column, label list)
    ("RISK   (risk level)", "gt_risk",   "pred_risk",   ["Expected", "Normal", "Taboo"]),
    ("TOPIC  (topic)",      "gt_topic",  "pred_topic",
        ["Religion", "Gender", "Race/Ethnicity", "Language/Wordplay", "History/Politics", "None"]),
    ("INTENT (intent)",     "gt_intent", "pred_intent", ["Intentional", "Unintentional", "Unclear"]),
]


def load_data():
    """Find the CSV file and read it into a table (DataFrame)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, PRED_FILE)
    if not os.path.exists(path):
        path = PRED_FILE  # fall back to the current folder if run elsewhere
    df = pd.read_csv(path)

    # [One important gotcha]
    # The topic label "None" turns into an empty cell (NaN) when a CSV is
    # saved and reloaded. If we leave it, the scores come out wrong, so we
    # restore the empty cells back to the string "None".
    for col in ["gt_topic", "pred_topic"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    return df


def score_one(y_true, y_pred, labels):
    """Compare the truth list (y_true) with the prediction list (y_pred)."""
    n = len(y_true)

    # (a) accuracy = number correct / total
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0

    # (b) per-class precision / recall / F1
    per_class = {}
    f1_list = []
    for lab in labels:
        # TP: actually lab AND predicted lab (correctly caught)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        # FP: not actually lab but predicted lab (false alarm)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        # FN: actually lab but predicted something else (missed)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        support = sum(1 for t in y_true if t == lab)  # how many truths had this label
        per_class[lab] = (precision, recall, f1, support)
        f1_list.append(f1)

    # macro-F1 = plain average of every class's F1 (each class weighted equally)
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return accuracy, macro_f1, per_class


def confusion_matrix(y_true, y_pred, labels):
    """Build a table counting actual (rows) vs predicted (columns)."""
    # If predictions contain values outside the label list (e.g. INVALID), add them as columns.
    extra = [x for x in dict.fromkeys(y_pred) if x not in labels]
    cols = labels + extra
    table = {a: {b: 0 for b in cols} for a in labels}
    for t, p in zip(y_true, y_pred):
        if t in table:
            table[t][p] = table[t].get(p, 0) + 1
    return table, cols


def short(label):
    """Shorten long labels so the table does not overflow."""
    return {
        "Language/Wordplay": "Lang/Word", "Race/Ethnicity": "Race/Eth",
        "History/Politics": "Hist/Pol", "Unintentional": "Uninten",
        "Intentional": "Inten", "Expected": "Expect",
    }.get(label, label)


def main():
    df = load_data()
    print("=" * 70)
    print(f"  Results analysis  ({len(df)} rows, versions: {list(df['version'].unique())})")
    print("=" * 70)

    for name, gt, pr, labels in TASKS:
        print("\n" + "#" * 70)
        print(f"#  TASK: {name}")
        print("#" * 70)

        for v in ["A", "B", "C"]:
            sub = df[df["version"] == v]
            y_true = list(sub[gt])
            y_pred = list(sub[pr])

            acc, macro, per_class = score_one(y_true, y_pred, labels)
            print(f"\n  -- Version {v} :  accuracy={acc:.3f}   macro-F1={macro:.3f} --")

            # per-class scores
            print(f"    {'class':<18}{'P':>6}{'R':>6}{'F1':>6}{'n':>6}")
            for lab in labels:
                p_, r_, f1, sup = per_class[lab]
                print(f"    {short(lab):<18}{p_:>6.2f}{r_:>6.2f}{f1:>6.2f}{sup:>6}")

            # confusion matrix
            table, cols = confusion_matrix(y_true, y_pred, labels)
            print("    confusion (rows = actual, cols = predicted):")
            header = "      " + "".join(f"{short(c)[:9]:>10}" for c in cols)
            print(header)
            for a in labels:
                row = f"    {short(a):<8}" + "".join(f"{table[a][c]:>10}" for c in cols)
                print(row)

    print("\n" + "=" * 70)
    print("  Done! The bigger the diagonal (actual == predicted), the better.")
    print("=" * 70)


if __name__ == "__main__":
    main()