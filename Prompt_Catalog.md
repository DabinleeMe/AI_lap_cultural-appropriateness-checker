# Prompt Catalog — Task 1 Step 2

This document explains the 3 prompt versions (A, B, C) used to test the LLM. The working code is in `prompts.py`. This document explains why each version is designed the way it is, and shows the full prompt text.

## Shared Design

All 3 versions must answer in the **same JSON format**. This way, Step 3 can read the answers from A, B, and C using one simple parser.

```json
{
  "risk": "Expected | Normal | Taboo",
  "topic": "Religion | Gender | Race/Ethnicity | Language/Wordplay | History/Politics | None",
  "intent": "Intentional | Unintentional | Unclear",
  "reason": "one sentence explanation"
}
```

This one JSON answer covers all **3 NLP tasks** from the project outline:

| Task from outline | JSON field |
|---|---|
| Risk classification (Expected/Normal/Taboo) | `risk` |
| Cultural topic classification | `topic` |
| Intent/context detection | `intent` (+ `reason`) |

Ground truth is stored in `Task1_Global_Advertising_Dataset.csv`, in the columns `gt_risk`, `gt_topic`, `gt_intent`, `gt_context`. `gt_intent` was created automatically from the first word of `gt_context` (Intentional/Unintentional).

**Why the topic list only has 6 categories**: The dataset also has more detailed labels, like "Culture-aware (Festival)". But this label means "handled well", not "a violation". Since the model needs to predict a *violated* topic, we simplified the list to 6 clean categories: Religion, Gender, Race/Ethnicity, Language/Wordplay, History/Politics, None.

**Note**: The 5 examples used in Prompt C (originally dataset rows with id 9, 21, 30, 35, 40) have been fully removed from `Task1_Global_Advertising_Dataset.csv`. They now only exist inside `prompts.py`, in `FEWSHOT_EXAMPLES`. So there is no risk of testing the model on an example it already saw — we don't need to manually exclude these rows anymore.

---

## Version A — Zero-shot (bare)

No persona. No examples. No label definitions. Just the label names and the output format. This is the simplest, most "naked" version.

```
Classify the marketing copy below.

Respond with ONLY a single valid JSON object. No markdown, no code fences, no extra text before or after.
The JSON object must have exactly these 4 keys:
{
  "risk": one of ["Expected", "Normal", "Taboo"],
  "topic": one of ["Religion", "Gender", "Race/Ethnicity", "Language/Wordplay", "History/Politics", "None"],
  "intent": one of ["Intentional", "Unintentional", "Unclear"],
  "reason": a single sentence (max 30 words) explaining the classification
}

Marketing copy: {text}
Target culture: {target_culture}
```

Hypothesis: without any label definitions, this version should get the lowest F1 score, especially for high-context topics like Language/Wordplay and Religion.

---

## Version B — System Role + Persona + Stakes

Gives the model an expert persona, adds pressure by saying "if you get this wrong, the brand loses real money", and tells the model not to judge from a Western viewpoint.

**System prompt:**
```
You are a veteran cross-border marketing consultant and cultural anthropologist with 20 years of experience advising global brands on international ad campaigns. You have lived and worked across East Asia, the Middle East, Latin America, and South Asia, and you are known for catching subtle cultural risks that Western-trained reviewers miss.

The stakes are real. If you misjudge this copy, the brand could face a viral backlash, boycotts, forced product recalls, and millions of dollars in losses -- the same kind of fallout as Dolce & Gabbana in China or Pepsi's Kendall Jenner ad. Your assessment will directly decide whether this campaign launches. A wrong call here is not a hypothetical error -- it is a costly real-world failure that damages the brand and your professional reputation.

Do NOT default to Western/American cultural assumptions. Evaluate strictly from the perspective of the target culture provided -- its religious norms, gender norms, historical sensitivities, and language-specific double meanings are the reference point, not your own.
```

**User prompt:** Same as Zero-shot (label definitions + output format + the text to classify). The only difference is that the persona is now in a separate system message.

---

## Version C — Few-shot + Chain-of-Thought

Uses the same persona as Version B, and adds two more things: (1) an instruction to reason step by step, and (2) 5 worked examples.

**System prompt:** Version B's system message, plus this:
```
For every new item, think step by step BEFORE giving your final answer:
Step 1: Identify any cultural element referenced in the copy (religion, gender roles, gestures, historical symbols, language/wordplay).
Step 2: Consider how someone from the target culture specifically (not a Western default) would perceive it.
Step 3: Decide the risk level and whether the reference was likely intentional or unintentional.
Show this reasoning under "Step-by-step:", then give the final JSON on its own line, prefixed with "FINAL:".
```

**5 few-shot examples** (originally dataset rows 9/21/30/35/40, chosen to cover different risk levels and topics. These rows no longer exist in the dataset — they now live only inside this prompt):

| id | target culture | risk | topic | why chosen |
|---|---|---|---|---|
| 30 | Global/Muslim-majority | Taboo | Religion | religious taboo example |
| 21 | UK | Taboo | Gender | gender taboo example |
| 9 | China | Taboo | Language/Wordplay | translation/wordplay taboo example |
| 35 | India | Normal | None | a festival handled well |
| 40 | USA | Expected | None | safe copy with no cultural content |

Each example is written as "Step-by-step" reasoning + "FINAL: {json}", so the model learns to answer new items in the same format. The full text of all 5 examples is in `prompts.py`, inside `FEWSHOT_EXAMPLES`.

---

## How this will be used

```python
from prompts import build_prompt

system_prompt, user_prompt = build_prompt("C", text=row["text"], target_culture=row["target_culture"])
# send system_prompt (skip if None) and user_prompt to Ollama/Gemini as-is
```

## Things to check later

- Does Llama 3.2 3B handle Version C's long few-shot prompt (about 5,500 characters) reliably? It's a small model, so a long prompt might make it lose track of the instructions. This is something to look at in Step 5 (error analysis).
- Track the JSON parsing failure rate as its own metric — Version A is the most likely to break the JSON format, since it has no examples to copy from.
