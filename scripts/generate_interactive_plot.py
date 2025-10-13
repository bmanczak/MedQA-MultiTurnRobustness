#!/usr/bin/env python3
"""Generate interactive bar chart showing model performance drops."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
import yaml

# Default follow-ups from config
DEFAULT_FOLLOWUPS = [
    "br_authority_prior",
    "br_autograder_prior",
    "br_commitment_alignment",
    "br_recency_prior",
    "br_social_proof_prior",
    "context_rag_style",
    "alternative_context",
    "edge_case_context",
]

# Human-readable names
FOLLOWUP_NAMES = {
    "br_authority_prior": "Authority Prior",
    "br_autograder_prior": "Autograder Prior",
    "br_commitment_alignment": "Commitment Alignment",
    "br_recency_prior": "Recency Prior",
    "br_social_proof_prior": "Social Proof Prior",
    "context_rag_style": "RAG-Style Context",
    "alternative_context": "Alternative Context",
    "edge_case_context": "Edge Case Context",
}

# Short model names (base)
MODEL_SHORT_NAMES = {
    "anthropic_claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic_claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "gemini_gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini_gemini-2.5-pro": "Gemini 2.5 Pro",
    "google_medgemma-4b-it": "MedGemma 4B",
    "google_medgemma-27b-it": "MedGemma 27B",
    "openai_gpt-4o-2024-08-06": "GPT-4o",
    "openai_gpt-5-2025-08-07": "GPT-5",
    "openai_gpt-5-mini-2025-08-07": "GPT-5 mini",
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "together_ai_openai_gpt-oss-120b": "GPT-OSS 120B",
    "together_ai_openai_gpt-oss-20b": "GPT-OSS 20B",
    "together_ai_Qwen_Qwen2.5-72B-Instruct": "Qwen2.5 72B",
    "together_ai_Qwen_Qwen2.5-72B-Instruct-Turbo": "Qwen2.5 72B Turbo",
    "xai_grok-4-fast-non-reasoning": "Grok 4 Fast",
    # Grok 4 (grok-4-0709) excluded (incomplete results)
}


def compute_accuracy_and_flips(jsonl_path: Path, baseline_path: Optional[Path] = None) -> tuple[float, float]:
    """
    Compute accuracy and flip rate from JSONL files.

    Args:
        jsonl_path: Path to JSONL file with predictions
        baseline_path: Path to baseline JSONL for computing flips

    Returns:
        Tuple of (accuracy %, flip rate %)
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    correct = 0
    total = 0
    predictions = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                idx = record.get("id")
                is_correct = record.get("correct", False)
                total += 1
                if is_correct:
                    correct += 1
                if idx is not None:
                    predictions[idx] = is_correct

    if total == 0:
        raise ValueError(f"Empty file: {jsonl_path}")

    accuracy = (correct / total) * 100
    flip_rate = 0.0

    # Compute flip rate if baseline provided
    if baseline_path and baseline_path.exists():
        baseline_preds = {}
        with open(baseline_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    idx = record.get("id")
                    is_correct = record.get("correct", False)
                    if idx is not None:
                        baseline_preds[idx] = is_correct

        # Count flips
        flips = 0
        common_ids = set(predictions.keys()) & set(baseline_preds.keys())
        for idx in common_ids:
            if predictions[idx] != baseline_preds[idx]:
                flips += 1

        if len(common_ids) > 0:
            flip_rate = (flips / len(common_ids)) * 100

    return accuracy, flip_rate


def extract_model_name(dir_name: str) -> str:
    """Extract model ID from directory name."""
    return dir_name.split("__")[0]


def load_model_config(model_dir: Path) -> Dict:
    """
    Load model configuration from resolved_config.yaml.

    Args:
        model_dir: Path to model results directory

    Returns:
        Dictionary with config parameters
    """
    config_path = model_dir / "resolved_config.yaml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract relevant parameters
        params = {}
        if config and "model" in config:
            model_cfg = config["model"]
            # Check for reasoning_effort
            if "extra_kwargs" in model_cfg and "reasoning_effort" in model_cfg["extra_kwargs"]:
                params["reasoning_effort"] = model_cfg["extra_kwargs"]["reasoning_effort"]

        return params
    except Exception:
        return {}


def create_model_label(model_id: str, config: Dict) -> str:
    """
    Create descriptive model label with configuration.

    Args:
        model_id: Base model identifier
        config: Configuration dictionary

    Returns:
        Human-readable model label
    """
    base_name = MODEL_SHORT_NAMES.get(model_id, model_id)

    # Add reasoning effort if present
    if "reasoning_effort" in config:
        return f"{base_name} ({config['reasoning_effort']})"

    return base_name


def process_results_directory(results_dir: Path) -> List[Dict]:
    """
    Process all results and compute accuracies with flip rates.

    Args:
        results_dir: Path to results directory

    Returns:
        List of dicts with model performance data
    """
    all_data = []

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_id = extract_model_name(model_dir.name)

        # Skip Grok 4 (incomplete results), but allow Grok 4 Fast
        if "grok-4-0709" in model_id.lower():
            continue
        first_turn_path = model_dir / "first_turn.jsonl"

        if not first_turn_path.exists():
            continue

        # Load configuration
        config = load_model_config(model_dir)
        model_label = create_model_label(model_id, config)

        try:
            baseline_acc, _ = compute_accuracy_and_flips(first_turn_path)
        except (FileNotFoundError, ValueError):
            continue

        followups_dir = model_dir / "followups"
        if not followups_dir.exists():
            continue

        for followup in DEFAULT_FOLLOWUPS:
            followup_path = followups_dir / f"{followup}.jsonl"
            if followup_path.exists():
                try:
                    followup_acc, flip_rate = compute_accuracy_and_flips(followup_path, first_turn_path)
                    all_data.append(
                        {
                            "model_id": model_id,
                            "model_name": model_label,
                            "model_dir": model_dir.name,
                            "followup": followup,
                            "followup_name": FOLLOWUP_NAMES[followup],
                            "baseline_acc": baseline_acc,
                            "followup_acc": followup_acc,
                            "drop": -(baseline_acc - followup_acc),  # Negative for hanging bars
                            "flip_rate": flip_rate,
                        }
                    )
                except (FileNotFoundError, ValueError):
                    continue

    return all_data


def create_interactive_plot(data: List[Dict], output_path: Path) -> None:
    """
    Create interactive bar chart with SE bars and flip rates on hover.

    Args:
        data: Performance data
        output_path: Output HTML path
    """
    # Compute statistics per model
    model_stats = {}
    for record in data:
        model = record["model_name"]
        if model not in model_stats:
            model_stats[model] = {"drops": [], "flip_rates": []}
        model_stats[model]["drops"].append(record["drop"])
        model_stats[model]["flip_rates"].append(record["flip_rate"])

    avg_stats = []
    for model, values in model_stats.items():
        drops = values["drops"]
        mean_drop = statistics.mean(drops)
        std_drop = statistics.stdev(drops) if len(drops) > 1 else 0
        avg_flip = statistics.mean(values["flip_rates"])

        avg_stats.append(
            {
                "model": model,
                "drop": mean_drop,
                "std": std_drop,
                "flip_rate": avg_flip,
            }
        )

    # Sort by drop (most negative first = worst performers)
    avg_stats.sort(key=lambda x: x["drop"])

    # Find max drop for y-axis range
    all_drops = [s["drop"] for s in avg_stats]
    max_abs_drop = max(abs(min(all_drops)), abs(max(all_drops)))

    # Create figure
    fig = go.Figure()

    # Add average bars (default view)
    fig.add_trace(
        go.Bar(
            x=[s["model"] for s in avg_stats],
            y=[s["drop"] for s in avg_stats],
            text=[f"{s['drop']:.1f}<br>(±{s['std']:.1f})" for s in avg_stats],
            textposition="inside",  # Position text inside bar, stacked
            textfont=dict(size=12),
            marker=dict(
                color="rgba(231, 111, 81, 0.1)",  # Very light fill
                line=dict(color="#E76F51", width=2.5),  # Thick border
            ),
            name="Average",
            visible=True,
            customdata=[[s["std"], s["flip_rate"]] for s in avg_stats],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Average Drop: %{y:.1f}%<br>"
                "Std Dev: %{customdata[0]:.1f}%<br>"
                "Avg Flip Rate: %{customdata[1]:.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add bars for each followup (hidden by default)
    for followup in DEFAULT_FOLLOWUPS:
        followup_data = [d for d in data if d["followup"] == followup]
        followup_data.sort(key=lambda x: x["drop"])

        fig.add_trace(
            go.Bar(
                x=[d["model_name"] for d in followup_data],
                y=[d["drop"] for d in followup_data],
                text=[f"{d['drop']:.1f}" for d in followup_data],
                textposition="inside",  # Position text inside bar
                textfont=dict(size=12),
                marker=dict(
                    color="rgba(231, 111, 81, 0.1)",
                    line=dict(color="#E76F51", width=2.5),
                ),
                name=FOLLOWUP_NAMES[followup],
                visible=False,
                customdata=[[d["flip_rate"]] for d in followup_data],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{FOLLOWUP_NAMES[followup]}<br>"
                    "Drop: %{y:.1f}%<br>"
                    "Flip Rate: %{customdata[0]:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Create dropdown menu
    buttons = [
        {
            "label": "Average (All 8 Interventions)",
            "method": "update",
            "args": [
                {"visible": [True] + [False] * len(DEFAULT_FOLLOWUPS)},
                {
                    "yaxis.title.text": "Accuracy Drop (%)",
                },
            ],
        }
    ]

    for i, followup in enumerate(DEFAULT_FOLLOWUPS):
        visible = [False] * (len(DEFAULT_FOLLOWUPS) + 1)
        visible[i + 1] = True

        buttons.append(
            {
                "label": FOLLOWUP_NAMES[followup],
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "yaxis.title.text": "Accuracy Drop (%)",
                    },
                ],
            }
        )

    # Update layout with inverted y-axis (no title, will be in caption)
    fig.update_layout(
        xaxis={
            "title": "",  # No x-axis label
            "tickangle": -45,
            "title_font": {"size": 14},
        },
        yaxis={
            "title": "Accuracy Drop (%)",
            "gridcolor": "#ECF0F1",
            "title_font": {"size": 14},
            "autorange": "reversed",  # Inverted: 0 at top
            "range": [2, -(max_abs_drop * 1.15)],  # Small positive to max negative
            "zeroline": True,
            "zerolinecolor": "#2C3E50",
            "zerolinewidth": 2,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
                "bgcolor": "#ECF0F1",
                "bordercolor": "#2C3E50",
                "font": {"size": 11},
            }
        ],
        width=1200,
        height=600,
        margin=dict(l=80, r=50, t=80, b=180),  # Increased bottom margin for larger caption
        annotations=[
            dict(
                text=(
                    "The performance of all state-of-the-art models on MedQA-MultiTurnRobustness drops.<br>"
                    "See the <a href='https://bmanczak.github.io/medqa_deep_robustness/'>paper</a> for the factors that impact the performance drops in more depth. "
                    "Bars hang downward showing negative accuracy change. Values show mean ± std dev across 8 interventions. "
                    "Reasoning/thinking parameters are explicitly mentioned when used (e.g., 'low', 'high'). Hover for flip rates."
                ),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.27,
                xanchor="center",
                yanchor="top",
                font=dict(size=13, color="#7f8c8d"),  # Larger gray italic caption
                align="center",
            )
        ],
    )

    fig.write_html(
        output_path,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )
    print(f"Interactive plot saved to: {output_path}")


def main() -> None:
    """Main execution."""
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    print("Processing results...")
    data = process_results_directory(results_dir)

    if not data:
        raise RuntimeError(
            "No evaluation data found in results/. Ensure prediction artifacts are present "
            "and Git LFS objects are fetched (e.g., `git lfs install` followed by `git lfs pull`)."
        )

    unique_models = len(set(d["model_name"] for d in data))
    print(f"Processed {len(data)} data points from {unique_models} model configurations")

    print("Creating interactive plot...")
    output_path = docs_dir / "chart.html"
    create_interactive_plot(data, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
