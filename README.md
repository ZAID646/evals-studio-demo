---
title: Evals Studio Demo
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.36.2
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Evals Studio Demo

Multi-provider benchmark comparison tool. Run evaluations across models and compare accuracy, F1, and ROUGE-L scores.

## Benchmarks

| Benchmark | Description | Samples |
|---|---|---|
| `mmlu` | General knowledge questions | 8 |
| `factual_consistency` | Summarization consistency | 3 |
| `instruction_following` | Exact instruction adherence | 3 |

## Metrics

- **Accuracy**: Exact string match (case-insensitive)
- **F1 Score**: Token-level precision/recall
- **ROUGE-L**: Longest Common Subsequence based recall

## API Keys

| Secret | Value |
|---|---|
| `CEREBRAS_API_KEY` | Your Cerebras API key |
| `NVIDIA_API_KEY` | Your NVIDIA API key |
