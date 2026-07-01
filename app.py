import json
import os
import re
import numpy as np
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

sns.set_style("whitegrid")

PROVIDERS = {
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "models": ["gpt-oss-120b"],
    },
    "NVIDIA": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "models": ["minimaxai/minimax-m3"],
    },
}

BENCHMARKS = {
    "mmlu": [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "What is 2 + 2?", "expected": "4"},
        {"question": "Who wrote Romeo and Juliet?", "expected": "William Shakespeare"},
        {"question": "What is the chemical symbol for water?", "expected": "H2O"},
        {"question": "Which planet is known as the Red Planet?", "expected": "Mars"},
        {"question": "What is the largest ocean on Earth?", "expected": "Pacific Ocean"},
        {"question": "Who painted the Mona Lisa?", "expected": "Leonardo da Vinci"},
        {"question": "What is the speed of light in m/s?", "expected": "299,792,458"},
    ],
    "factual_consistency": [
        {"question": "Summarize: The Eiffel Tower is in Paris.", "expected": "The Eiffel Tower is located in Paris."},
        {"question": "Summarize: Water boils at 100 degrees Celsius at sea level.", "expected": "Water boils at 100°C at sea level."},
        {"question": "Summarize: The human heart has four chambers.", "expected": "The human heart consists of four chambers."},
    ],
    "instruction_following": [
        {"question": "Respond with only the word 'hello'.", "expected": "hello"},
        {"question": "List exactly three colors separated by commas.", "expected": "red, blue, green"},
        {"question": "Count from 1 to 5, separated by spaces.", "expected": "1 2 3 4 5"},
    ],
}


def accuracy_score(predictions, references):
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    return correct / len(predictions)


def f1_score(predictions, references):
    scores = []
    for p, r in zip(predictions, references):
        p_tokens = set(p.lower().split())
        r_tokens = set(r.lower().split())
        if not r_tokens:
            scores.append(1.0 if not p_tokens else 0.0)
            continue
        if not p_tokens:
            scores.append(0.0)
            continue
        precision = len(p_tokens & r_tokens) / len(p_tokens)
        recall = len(p_tokens & r_tokens) / len(r_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(predictions, references):
    scores = []
    for p, r in zip(predictions, references):
        p_tokens = p.lower().split()
        r_tokens = r.lower().split()
        lcs = lcs_length(p_tokens, r_tokens)
        if len(r_tokens) == 0:
            scores.append(1.0 if len(p_tokens) == 0 else 0.0)
        else:
            prec = lcs / len(p_tokens) if p_tokens else 0
            rec = lcs / len(r_tokens)
            if prec + rec == 0:
                scores.append(0.0)
            else:
                scores.append(2 * prec * rec / (prec + rec))
    return float(np.mean(scores)) if scores else 0.0


def run_evaluation(provider_name, model, benchmark_name, progress=gr.Progress()):
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return None, f"Unknown provider: {provider_name}"

    api_key = os.environ.get(provider["api_key_env"], "")
    if not api_key:
        return None, f"❌ **{provider_name} API key not set.** Configure `{provider['api_key_env']}` as a Space secret."

    client = OpenAI(api_key=api_key, base_url=provider["base_url"])
    samples = BENCHMARKS.get(benchmark_name, [])
    if not samples:
        return None, f"Unknown benchmark: {benchmark_name}"

    predictions = []
    references = [s["expected"] for s in samples]

    progress(0, desc=f"Starting evaluation on {benchmark_name}...")

    for i, sample in enumerate(samples):
        progress(
            (i + 1) / len(samples),
            desc=f"Sample {i+1}/{len(samples)}: {sample['question'][:60]}...",
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": sample["question"]}],
                temperature=0,
                max_tokens=150,
            )
            pred = response.choices[0].message.content.strip()
        except Exception as e:
            pred = f"[ERROR: {e}]"

        predictions.append(pred)

    acc = accuracy_score(predictions, references)
    f1 = f1_score(predictions, references)
    rl = rouge_l_score(predictions, references)

    results = {
        "model": f"{provider_name}/{model}",
        "benchmark": benchmark_name,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "rougeL": round(rl, 4),
        "samples": [
            {"question": s["question"], "expected": s["expected"], "predicted": p}
            for s, p in zip(samples, predictions)
        ],
    }

    return results, None


def update_models(provider_name):
    provider = PROVIDERS.get(provider_name)
    if provider:
        models = provider["models"]
        return gr.Dropdown(choices=models, value=models[0])
    return gr.Dropdown(choices=[], value=None)


def generate_chart(results_list):
    if not results_list:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Run an evaluation to see results", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["accuracy", "f1", "rougeL"]
    n_models = len(results_list)
    width = 0.8 / n_models
    x = np.arange(len(metrics))
    colors = sns.color_palette("husl", n_models)

    for i, r in enumerate(results_list):
        values = [r[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=r["model"], color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(["Accuracy", "F1 Score", "ROUGE-L"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Benchmark Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def on_run(provider, model, benchmark, state):
    results, error = run_evaluation(provider, model, benchmark)
    if error:
        return state, error, None, gr.DataFrame(value=[], headers=[]), gr.DataFrame(value=[], headers=[])

    state.append(results)

    fig = generate_chart(state)

    summary_data = [[r["model"], r["benchmark"], r["accuracy"], r["f1"], r["rougeL"]] for r in state]
    summary_headers = ["Model", "Benchmark", "Accuracy", "F1", "ROUGE-L"]

    samples = results["samples"]
    samples_data = [
        [s["question"][:80], s["expected"][:80], s["predicted"][:80]]
        for s in samples
    ]

    return state, None, fig, gr.DataFrame(value=summary_data, headers=summary_headers), gr.DataFrame(
        value=samples_data,
        headers=["Question", "Expected", "Predicted"],
    )


def on_clear():
    return [], None, None, gr.DataFrame(value=[], headers=[]), gr.DataFrame(value=[], headers=[])


with gr.Blocks(
    title="Evals Studio Demo",
    theme=gr.themes.Soft(),
    fill_height=True,
) as demo:
    gr.Markdown(
        "# 📊 Evals Studio Demo\n"
        "### Multi-Provider Benchmark Comparison powered by **Cerebras** & **NVIDIA**\n\n"
        "Run benchmarks across models and compare accuracy, F1, and ROUGE-L scores."
    )

    state = gr.State([])

    with gr.Row():
        provider_dd = gr.Dropdown(
            choices=list(PROVIDERS.keys()),
            value=list(PROVIDERS.keys())[0],
            label="AI Provider",
            scale=1,
        )
        model_dd = gr.Dropdown(
            choices=PROVIDERS[list(PROVIDERS.keys())[0]]["models"],
            value=PROVIDERS[list(PROVIDERS.keys())[0]]["models"][0],
            label="Model",
            scale=1,
        )
        benchmark_dd = gr.Dropdown(
            choices=list(BENCHMARKS.keys()),
            value=list(BENCHMARKS.keys())[0],
            label="Benchmark",
            scale=1,
        )

    provider_dd.change(fn=update_models, inputs=provider_dd, outputs=model_dd)

    with gr.Row():
        run_btn = gr.Button("▶ Run Evaluation", variant="primary", scale=1)
        clear_btn = gr.Button("🗑 Clear Results", variant="secondary", scale=0)

    error_msg = gr.Markdown(visible=True)

    plot = gr.Plot(label="Results Comparison", show_label=True)

    summary_table = gr.DataFrame(
        label="Summary",
        headers=["Model", "Benchmark", "Accuracy", "F1", "ROUGE-L"],
        datatype=["str", "str", "number", "number", "number"],
    )

    with gr.Accordion("📝 Sample Predictions", open=False):
        samples_table = gr.DataFrame(
            headers=["Question", "Expected", "Predicted"],
            datatype=["str", "str", "str"],
        )

    run_btn.click(
        fn=on_run,
        inputs=[provider_dd, model_dd, benchmark_dd, state],
        outputs=[state, error_msg, plot, summary_table, samples_table],
    )

    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[state, error_msg, plot, summary_table, samples_table],
    )

if __name__ == "__main__":
    demo.launch()
