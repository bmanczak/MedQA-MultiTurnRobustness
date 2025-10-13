#!/usr/bin/env python3
"""Generate SVG preview of the bar chart."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from generate_interactive_plot import process_results_directory


def create_svg_preview(data: List[Dict], output_path: Path) -> None:
    """
    Create SVG preview of bar chart with inverted y-axis.

    Args:
        data: Performance data
        output_path: Output SVG path
    """
    # Filter out Grok 4 (incomplete results), but allow Grok 4 Fast
    # Note: Grok 4 (grok-4-0709) is already filtered in process_results_directory

    # Compute statistics per model
    model_stats = {}
    for record in data:
        model = record["model_name"]
        if model not in model_stats:
            model_stats[model] = {"drops": []}
        model_stats[model]["drops"].append(record["drop"])

    avg_stats = []
    for model, values in model_stats.items():
        drops = values["drops"]
        mean_drop = statistics.mean(drops)
        std_drop = statistics.stdev(drops) if len(drops) > 1 else 0

        avg_stats.append({"model": model, "drop": mean_drop, "std": std_drop})

    # Sort by drop (most negative first)
    avg_stats.sort(key=lambda x: x["drop"])

    # SVG dimensions
    width = 1200
    height = 600
    margin = {"top": 80, "right": 50, "bottom": 150, "left": 80}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    # Find range for y-axis (inverted: 0 at top, negative below)
    all_drops = [s["drop"] for s in avg_stats]
    min_drop = min(all_drops)
    max_drop = max(all_drops)
    y_max = 2  # Start slightly positive
    y_min = min(min_drop * 1.15, -5)  # Ensure we show at least -5%

    # Bar calculations
    n_bars = len(avg_stats)
    bar_group_width = plot_width / n_bars
    bar_width = min(bar_group_width * 0.7, 60)
    bar_spacing = bar_group_width

    def scale_y(val: float) -> float:
        """Scale y value (inverted: 0 at top, negatives below)."""
        normalized = (val - y_max) / (y_min - y_max)
        return margin["top"] + normalized * plot_height

    # Start SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "  <defs>",
        "    <style>",
        "      .grid-line { stroke: #ECF0F1; stroke-width: 1; }",
        "      .axis-line { stroke: #2C3E50; stroke-width: 2; }",
        "      .zero-line { stroke: #2C3E50; stroke-width: 2; }",
        "      .bar { fill: rgba(231, 111, 81, 0.1); stroke: #E76F51; stroke-width: 2.5; }",
        "      .error-bar { stroke: #2C3E50; stroke-width: 1.5; }",
        "      .label { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 11px; fill: #2C3E50; }",
        "      .value-label { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 14px; fill: #2C3E50; font-weight: 500; }",
        "      .axis-label { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 14px; fill: #2C3E50; }",
        "      .title { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 20px; fill: #2C3E50; font-weight: 600; }",
        "      .caption { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 11px; fill: #7f8c8d; font-style: italic; }",
        "    </style>",
        "  </defs>",
        "  ",
        "  <!-- Background -->",
        '  <rect width="100%" height="100%" fill="white"/>',
        "  ",
        "  <!-- Grid lines -->",
    ]

    # Add horizontal grid lines (every 5% from 0 downward)
    for i in range(int(y_max), int(y_min) - 1, -5):
        y = scale_y(i)
        svg_parts.append(
            f'  <line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" class="grid-line"/>'
        )
        svg_parts.append(f'  <text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" class="label">{i}</text>')

    # Zero line (thicker)
    zero_y = scale_y(0)
    svg_parts.append(
        f'  <line x1="{margin["left"]}" y1="{zero_y}" x2="{width - margin["right"]}" y2="{zero_y}" class="zero-line"/>'
    )

    # Add axes
    svg_parts.extend(
        [
            "  ",
            "  <!-- Axes -->",
            f'  <line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height - margin["bottom"]}" class="axis-line"/>',
            f'  <line x1="{margin["left"]}" y1="{height - margin["bottom"]}" x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" class="axis-line"/>',
            "  ",
            "  <!-- Axis labels -->",
            f'  <text x="{30}" y="{height/2}" text-anchor="middle" class="axis-label" transform="rotate(-90, 30, {height/2})">Accuracy Drop (%)</text>',
            "  ",
            "  <!-- Bars -->",
        ]
    )

    # Add bars and error bars
    for i, record in enumerate(avg_stats):
        x = margin["left"] + (i * bar_spacing) + (bar_group_width - bar_width) / 2
        bar_top_y = scale_y(0)  # Bars hang from 0
        bar_bottom_y = scale_y(record["drop"])
        bar_height = abs(bar_bottom_y - bar_top_y)

        # Bar (hanging downward from 0)
        svg_parts.append(f'  <rect x="{x}" y="{bar_top_y}" width="{bar_width}" height="{bar_height}" class="bar">')
        svg_parts.append(f'    <title>{record["model"]}: {record["drop"]:.1f}% drop (±{record["std"]:.1f}%)</title>')
        svg_parts.append("  </rect>")

        # Value labels stacked (drop on top, std below)
        label_x = x + bar_width / 2
        label_y_drop = bar_bottom_y - 18  # Position for drop value
        label_y_std = bar_bottom_y - 4  # Position for std value (below drop)
        svg_parts.append(
            f'  <text x="{label_x}" y="{label_y_drop}" text-anchor="middle" class="value-label">{record["drop"]:.1f}</text>'
        )
        svg_parts.append(
            f'  <text x="{label_x}" y="{label_y_std}" text-anchor="middle" class="value-label" font-size="11">(±{record["std"]:.1f})</text>'
        )

        # Model name label (rotated)
        label_x = x + bar_width / 2
        label_y_pos = height - margin["bottom"] + 15
        svg_parts.append(
            f'  <text x="{label_x}" y="{label_y_pos}" text-anchor="end" class="label" transform="rotate(-45, {label_x}, {label_y_pos})">{record["model"]}</text>'
        )

    # Caption (gray italic style)
    caption_y_start = height - 40
    svg_parts.extend(
        [
            "  ",
            "  <!-- Caption -->",
            f'  <text x="{width/2}" y="{caption_y_start}" text-anchor="middle" class="caption">',
            "    The performance of all state-of-the-art models on MedQA-MultiTurnRobustness drops.",
            "  </text>",
            f'  <text x="{width/2}" y="{caption_y_start + 16}" text-anchor="middle" class="caption">',
            "    See the paper for the factors that impact the performance drops in more depth. Bars hang downward showing negative accuracy change.",
            "  </text>",
            f'  <text x="{width/2}" y="{caption_y_start + 30}" text-anchor="middle" class="caption">',
            "    Values show mean ± std dev across 8 interventions. Reasoning parameters mentioned when used.",
            "  </text>",
        ]
    )

    svg_parts.append("</svg>")

    output_path.write_text("\n".join(svg_parts))
    print(f"SVG preview saved to: {output_path}")


def main() -> None:
    """Main execution."""
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    print("Processing results...")
    data = process_results_directory(results_dir)
    print(f"Processed {len(data)} data points")

    print("Creating SVG preview...")
    output_path = docs_dir / "preview.svg"
    create_svg_preview(data, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
