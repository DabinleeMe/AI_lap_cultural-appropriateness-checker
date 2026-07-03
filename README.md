# Cultural Appropriateness Checker 🚧 (Work in Progress)

> Module 1 of a Cultural Intelligence Platform for global content teams —
> an A/B/C prompt-engineering evaluation for detecting cultural risk in global ad copy.

**Status:** 🔨 Active development (M.Sc. AI in Business, SRH Berlin). 
Evaluation pipeline and prompt catalog are functional; scoring and analysis in progress.

## Why this exists

AI now produces marketing content at unprecedented scale — but checking
whether that content *fits the target culture* is still slow, manual,
and inconsistent. It sits with human consultants, focus groups, and
last-minute localization QA. And because most LLMs lean heavily toward
English and Western norms, they quietly miss the cultural nuance of
everywhere else.

This project asks: can an LLM, with the right prompt design, flag
cultural risk at the *draft* stage — as a measurable risk score, not a
verdict? Not to replace local experts, but to act as a triage layer so
they only review the genuinely ambiguous cases.

## What it does (Module 1 of 4)

This repo is **Task 1 of a four-module Cultural Intelligence Platform**
(M.Sc. AI in Business, SRH Berlin):

1. **NLP (this repo)** — prompt-based cultural risk classification
2. **Analytics** — dashboard mapping where the LLM fails, by culture × norm
3. **RAG** — grounding with NormBank / CultureBank to cut hallucination
4. **Agent** — LangGraph agent mimicking the human review flow, with a
   human-in-the-loop gate

**This module** evaluates three prompt strategies — zero-shot, few-shot,
and persona — on 84 real-world samples (actual global ad controversies +
normal marketing copy) across 8 cultures selected along the Hofstede
Uncertainty Avoidance spectrum (China 30 → Japan 92). Each sample is
classified on three axes: risk level (Expected / Normal / Taboo),
violated cultural topic, and intent. Metrics: accuracy and macro F1.

## Roadmap

- [x] Module 1 — Prompt-based risk classification (this repo)
- [ ] Module 2 — Cultural risk dashboard (prediction logs × Hofstede 6D)
- [ ] Module 3 — RAG grounding with NormBank / CultureBank (ChromaDB)
- [ ] Module 4 — Agentic review workflow (LangGraph, human-in-the-loop)
