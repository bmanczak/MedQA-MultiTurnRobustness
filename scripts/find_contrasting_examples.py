#!/usr/bin/env python3
"""
Find contrasting examples where one model flips (correct → incorrect)
and another model doesn't (correct → correct) on the same question.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Categorize followups
STATIC_FOLLOWUPS = [
    "br_authority_prior",
    "br_autograder_prior",
    "br_commitment_alignment",
    "br_recency_prior",
    "br_social_proof_prior",
]

DYNAMIC_FOLLOWUPS = [
    "context_rag_style",
    "alternative_context",
    "edge_case_context",
]

MODEL_NAMES = {
    "anthropic_claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic_claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "gemini_gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini_gemini-2.5-pro": "Gemini 2.5 Pro",
    "google_medgemma-4b-it": "MedGemma 4B",
    "google_medgemma-27b-it": "MedGemma 27B",
    "openai_gpt-4o-2024-08-06": "GPT-4o",
    "openai_gpt-5-2025-08-07": "GPT-5",
    "openai_gpt-5-mini-2025-08-07": "GPT-5 mini",
    "together_ai_openai_gpt-oss-120b": "GPT-OSS 120B",
    "together_ai_openai_gpt-oss-20b": "GPT-OSS 20B",
    "together_ai_Qwen_Qwen2.5-72B-Instruct-Turbo": "Qwen2.5 72B Turbo",
    "xai_grok-4-fast-non-reasoning": "Grok 4 Fast",
}

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


def load_model_data(results_dir: Path) -> Dict:
    """
    Load all model results into a structured dictionary.

    Returns:
        Dict mapping model_id -> question_id -> followup_type -> data
    """
    all_data = {}

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name.split("__")[0]

        if model_id not in MODEL_NAMES:
            continue

        # Load first turn data
        first_turn_path = model_dir / "first_turn.jsonl"
        if not first_turn_path.exists():
            continue

        first_turn_data = {}
        with open(first_turn_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    first_turn_data[record["id"]] = record

        # Load followup data
        followups_dir = model_dir / "followups"
        if not followups_dir.exists():
            continue

        model_results = {}

        for followup_file in followups_dir.glob("*.jsonl"):
            followup_type = followup_file.stem

            with open(followup_file) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        question_id = record["id"]

                        if question_id not in model_results:
                            model_results[question_id] = {}

                        # Combine first turn and followup data
                        first_turn = first_turn_data.get(question_id, {})
                        model_results[question_id][followup_type] = {
                            "first_turn": first_turn,
                            "followup": record,
                        }

        all_data[model_id] = model_results

    return all_data


def has_substantial_reasoning(text: str) -> bool:
    """
    Check if reasoning text contains substantial step-by-step thinking.

    Filters out minimal responses like just "Final Answer: (A)".
    """
    if not text:
        return False

    # Must be longer than just a final answer
    if len(text) < 100:
        return False

    # Should contain some reasoning indicators
    reasoning_indicators = [
        "step",
        "because",
        "therefore",
        "since",
        "analysis",
        "consider",
        "summary",
        "suggests",
        "indicates",
        "reasoning",
        "let me",
        "let's",
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in reasoning_indicators)


def get_model_family(model_id: str) -> str:
    """Extract model family from model ID."""
    if "claude-sonnet-4-5" in model_id:
        return "claude-4.5"
    elif "claude-sonnet-4" in model_id:
        return "claude-4"
    elif "anthropic" in model_id or "claude" in model_id:
        return "anthropic"
    elif "grok" in model_id or "xai" in model_id:
        return "grok"
    elif "medgemma-27b" in model_id:
        return "medgemma-27b"
    elif "medgemma-4b" in model_id:
        return "medgemma-4b"
    elif "medgemma" in model_id:
        return "medgemma"
    elif "gemini" in model_id:
        return "gemini"
    elif "gpt-oss-20b" in model_id:
        return "gpt-oss-20b"
    elif "gpt-oss-120b" in model_id:
        return "gpt-oss-120b"
    elif "gpt-oss" in model_id:
        return "gpt-oss"
    elif "gpt-5-mini" in model_id:
        return "gpt-5-mini"
    elif "gpt-5" in model_id:
        return "gpt-5"
    elif "gpt-4" in model_id:
        return "gpt-4"
    elif "qwen" in model_id.lower():
        return "qwen"
    else:
        return "other"


def calculate_text_length(example: Dict) -> int:
    """Calculate total text length of an example."""
    first = example["first_turn"]
    followup = example["followup"]

    length = (
        len(first.get("prompt") or "")
        + len(first.get("response") or "")
        + len(followup.get("prompt") or "")
        + len(followup.get("response") or "")
    )

    return length


def find_contrasting_pairs(all_data: Dict, followup_types: List[str]) -> List[Dict]:
    """
    Find pairs where one model flips and another doesn't.

    Args:
        all_data: All model data
        followup_types: List of followup types to search

    Returns:
        List of contrasting examples with metadata
    """
    contrasts = []

    models = list(all_data.keys())

    for followup_type in followup_types:
        # For each question, find models that have data
        question_model_map = {}

        for model_id in models:
            for question_id, followups in all_data[model_id].items():
                if followup_type in followups:
                    if question_id not in question_model_map:
                        question_model_map[question_id] = {}
                    question_model_map[question_id][model_id] = followups[followup_type]

        # Find contrasting pairs for each question
        for question_id, model_data in question_model_map.items():
            if len(model_data) < 2:
                continue

            # Find models that flip vs don't flip (with substantial reasoning)
            flippers = []
            non_flippers = []

            for model_id, data in model_data.items():
                first_correct = data["first_turn"].get("correct", False)
                followup_correct = data["followup"].get("correct", False)

                # Check for substantial reasoning in both turns
                first_reasoning = data["first_turn"].get("response", "")
                followup_reasoning = data["followup"].get("response", "")

                if not has_substantial_reasoning(first_reasoning):
                    continue
                if not has_substantial_reasoning(followup_reasoning):
                    continue

                if first_correct and not followup_correct:
                    flippers.append((model_id, data))
                elif first_correct and followup_correct:
                    non_flippers.append((model_id, data))

            # Create pairs
            for flipper_model, flipper_data in flippers:
                for non_flipper_model, non_flipper_data in non_flippers:
                    total_length = calculate_text_length(flipper_data) + calculate_text_length(non_flipper_data)

                    contrasts.append(
                        {
                            "question_id": question_id,
                            "followup_type": followup_type,
                            "flipper_model": flipper_model,
                            "non_flipper_model": non_flipper_model,
                            "flipper_data": flipper_data,
                            "non_flipper_data": non_flipper_data,
                            "text_length": total_length,
                        }
                    )

    return contrasts


def select_best_examples(
    contrasts: List[Dict], count: int, used_model_pairs: set = None, used_model_families: set = None
) -> List[Dict]:
    """
    Select best examples ensuring model family diversity, different interventions,
    and different questions. Prefer shorter examples within these constraints.

    Args:
        contrasts: List of contrast examples
        count: Number to select
        used_model_pairs: Set of already used model pairs
        used_model_families: Set of already used model families

    Returns:
        Selected examples
    """
    if used_model_pairs is None:
        used_model_pairs = set()
    if used_model_families is None:
        used_model_families = set()

    # Sort by text length
    contrasts_sorted = sorted(contrasts, key=lambda x: x["text_length"])

    selected = []
    used_interventions = set()
    used_questions = set()

    # Priority 1: Get examples with new model families
    for contrast in contrasts_sorted:
        flipper_family = get_model_family(contrast["flipper_model"])
        non_flipper_family = get_model_family(contrast["non_flipper_model"])

        model_pair = tuple(sorted([contrast["flipper_model"], contrast["non_flipper_model"]]))
        intervention = contrast["followup_type"]
        question = contrast["question_id"]

        # Prefer new model families, interventions, and questions
        has_new_family = flipper_family not in used_model_families or non_flipper_family not in used_model_families

        if (
            has_new_family
            and intervention not in used_interventions
            and question not in used_questions
            and model_pair not in used_model_pairs
        ):
            selected.append(contrast)
            used_model_pairs.add(model_pair)
            used_model_families.add(flipper_family)
            used_model_families.add(non_flipper_family)
            used_interventions.add(intervention)
            used_questions.add(question)

            if len(selected) >= count:
                break

    # Priority 2: Relax model family constraint if needed
    if len(selected) < count:
        for contrast in contrasts_sorted:
            if contrast in selected:
                continue

            flipper_family = get_model_family(contrast["flipper_model"])
            non_flipper_family = get_model_family(contrast["non_flipper_model"])

            intervention = contrast["followup_type"]
            question = contrast["question_id"]
            model_pair = tuple(sorted([contrast["flipper_model"], contrast["non_flipper_model"]]))

            # At least ensure different interventions and questions
            if intervention not in used_interventions and question not in used_questions:
                selected.append(contrast)
                used_interventions.add(intervention)
                used_questions.add(question)
                used_model_pairs.add(model_pair)
                used_model_families.add(flipper_family)
                used_model_families.add(non_flipper_family)

                if len(selected) >= count:
                    break

    return selected


def format_example_for_output(contrast: Dict) -> Dict:
    """Format a contrast example for YAML output."""
    flipper = contrast["flipper_data"]
    non_flipper = contrast["non_flipper_data"]

    # Extract question from first turn prompt
    first_prompt = flipper["first_turn"]["prompt"]
    question_start = first_prompt.find("Question:")
    response_start = first_prompt.find("Response (")
    question_text = first_prompt[question_start:response_start].strip() if question_start != -1 else ""

    return {
        "question_id": contrast["question_id"],
        "intervention": {
            "type": contrast["followup_type"],
            "name": FOLLOWUP_NAMES[contrast["followup_type"]],
        },
        "question": question_text,
        "model_flip": {
            "name": MODEL_NAMES[contrast["flipper_model"]],
            "id": contrast["flipper_model"],
            "first_turn": {
                "reasoning": flipper["first_turn"]["response"],
                "answer": flipper["first_turn"]["pred"],
                "correct": True,
            },
            "followup": {
                "prompt": flipper["followup"]["prompt"],
                "reasoning": flipper["followup"]["response"],
                "answer": flipper["followup"]["pred"],
                "correct": False,
            },
        },
        "model_no_flip": {
            "name": MODEL_NAMES[contrast["non_flipper_model"]],
            "id": contrast["non_flipper_model"],
            "first_turn": {
                "reasoning": non_flipper["first_turn"]["response"],
                "answer": non_flipper["first_turn"]["pred"],
                "correct": True,
            },
            "followup": {
                "prompt": non_flipper["followup"]["prompt"],
                "reasoning": non_flipper["followup"]["response"],
                "answer": non_flipper["followup"]["pred"],
                "correct": True,
            },
        },
        "gold_answer": flipper["first_turn"]["gold"],
        "text_length": contrast["text_length"],
    }


def main():
    """Main execution."""
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"

    print("Loading all model results...")
    all_data = load_model_data(results_dir)
    print(f"Loaded data for {len(all_data)} models")

    print("\nFinding contrasting examples for STATIC followups...")
    static_contrasts = find_contrasting_pairs(all_data, STATIC_FOLLOWUPS)
    print(f"Found {len(static_contrasts)} static contrasts")

    print("\nFinding contrasting examples for DYNAMIC followups...")
    dynamic_contrasts = find_contrasting_pairs(all_data, DYNAMIC_FOLLOWUPS)
    print(f"Found {len(dynamic_contrasts)} dynamic contrasts")

    print("\nSelecting best examples - one for each intervention type...")
    print("   Priority: Include MedGemma 27B, Claude 4 & 4.5, GPT-OSS 20B, GPT-5 Mini")
    print("   Constraint: GPT-5 as stable model max 2 times, diverse model pairings")

    # Goal: One example for each intervention type (8 total)
    # Maximize model diversity, prioritize requested models, limit GPT-5 favoritism

    used_model_ids = set()  # Track individual model IDs, not just families
    selected_examples = []
    stable_model_count = {}  # Track how many times each model is the "stable" one

    # Priority models to ensure we include
    priority_models = {
        "google_medgemma-27b-it",
        "anthropic_claude-sonnet-4-20250514",
        "anthropic_claude-sonnet-4-5-20250929",
        "together_ai_openai_gpt-oss-20b",
        "openai_gpt-5-mini-2025-08-07",
    }

    # Combine all contrasts
    all_contrasts = static_contrasts + dynamic_contrasts

    # Group by intervention type
    contrasts_by_type = {}
    for contrast in all_contrasts:
        intervention = contrast["followup_type"]
        if intervention not in contrasts_by_type:
            contrasts_by_type[intervention] = []
        contrasts_by_type[intervention].append(contrast)

    # Sort each group by length
    for intervention in contrasts_by_type:
        contrasts_by_type[intervention].sort(key=lambda x: x["text_length"])

    # For each intervention type, find the best example with model diversity
    all_intervention_types = STATIC_FOLLOWUPS + DYNAMIC_FOLLOWUPS

    for intervention_type in all_intervention_types:
        if intervention_type not in contrasts_by_type:
            print(f"   ⚠ No examples found for {intervention_type}")
            continue

        candidates = contrasts_by_type[intervention_type]

        # Find the best candidate that maximizes model diversity
        best_candidate = None
        best_score = float("-inf")  # Accept any score, even negative
        skipped_count = 0

        for candidate in candidates:
            flipper_model = candidate["flipper_model"]
            stable_model = candidate["non_flipper_model"]
            f1 = get_model_family(flipper_model)
            f2 = get_model_family(stable_model)

            # Skip if GPT-5 (not mini) is stable and already appears 2+ times
            if "gpt-5-2025" in stable_model and stable_model_count.get(stable_model, 0) >= 2:
                skipped_count += 1
                continue

            # Skip exact same pair
            pair = tuple(sorted([flipper_model, stable_model]))
            if pair in [(tuple(sorted([e["flipper_model"], e["non_flipper_model"]]))) for e in selected_examples]:
                skipped_count += 1
                continue

            # Score: prioritize new model families and priority models
            score = 0

            # Big bonus for priority models not yet used
            if flipper_model in priority_models and flipper_model not in used_model_ids:
                score += 25
            if stable_model in priority_models and stable_model not in used_model_ids:
                score += 25

            # Bonus for new individual models
            if flipper_model not in used_model_ids:
                score += 12
            if stable_model not in used_model_ids:
                score += 12

            # Penalty for overused stable models
            stable_count = stable_model_count.get(stable_model, 0)
            score -= stable_count * 7

            # Light penalty if model appears again (allows reuse to reach 8 examples)
            if flipper_model in used_model_ids:
                score -= 3
            if stable_model in used_model_ids:
                score -= 3

            # Secondary: shorter is better (less weight)
            score -= candidate["text_length"] / 15000

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate:
            selected_examples.append(best_candidate)
            stable_model = best_candidate["non_flipper_model"]
            flipper_model = best_candidate["flipper_model"]
            stable_model_count[stable_model] = stable_model_count.get(stable_model, 0) + 1

            used_model_ids.add(flipper_model)
            used_model_ids.add(stable_model)

            f1_name = MODEL_NAMES.get(flipper_model, flipper_model)
            f2_name = MODEL_NAMES.get(stable_model, stable_model)
            print(f"   ✓ {FOLLOWUP_NAMES[intervention_type]}: {f1_name} flips, {f2_name} stays")
        else:
            print(
                f"   ✗ {FOLLOWUP_NAMES[intervention_type]}: NO CANDIDATE FOUND (skipped {skipped_count}/{len(candidates)})"
            )

    # Split back into static and dynamic
    selected_static = [ex for ex in selected_examples if ex["followup_type"] in STATIC_FOLLOWUPS]
    selected_dynamic = [ex for ex in selected_examples if ex["followup_type"] in DYNAMIC_FOLLOWUPS]

    # Final check of coverage
    all_selected = selected_static + selected_dynamic
    covered_models = set()
    covered_interventions = set()
    stable_models_used = {}
    flipper_models_used = {}
    for ex in all_selected:
        covered_models.add(ex["flipper_model"])
        covered_models.add(ex["non_flipper_model"])
        covered_interventions.add(ex["followup_type"])
        stable = ex["non_flipper_model"]
        flipper = ex["flipper_model"]
        stable_models_used[stable] = stable_models_used.get(stable, 0) + 1
        flipper_models_used[flipper] = flipper_models_used.get(flipper, 0) + 1

    print(f"\n   Final covered models: {len(covered_models)}")
    print(f"   Final interventions: {len(covered_interventions)}/8")
    print(f"\n   Stable model usage (stays correct):")
    for model, count in sorted(stable_models_used.items(), key=lambda x: -x[1]):
        print(f"      {MODEL_NAMES.get(model, model)}: {count}x")
    print(f"\n   Flipper model usage (flips to incorrect):")
    for model, count in sorted(flipper_models_used.items(), key=lambda x: -x[1]):
        print(f"      {MODEL_NAMES.get(model, model)}: {count}x")

    # Format for output
    output = {
        "static_examples": [format_example_for_output(ex) for ex in selected_static],
        "dynamic_examples": [format_example_for_output(ex) for ex in selected_dynamic],
    }

    # Save to YAML
    output_path = repo_root / "docs" / "curated_examples.yaml"
    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"\n✓ Saved curated examples to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF SELECTED EXAMPLES")
    print("=" * 60)

    print("\n📌 STATIC EXAMPLES:")
    for i, ex in enumerate(selected_static, 1):
        flipper_family = get_model_family(ex["flipper_model"])
        non_flipper_family = get_model_family(ex["non_flipper_model"])
        print(f"\n  {i}. {FOLLOWUP_NAMES[ex['followup_type']]}")
        print(f"     Question ID: {ex['question_id']}")
        print(f"     Flips: {MODEL_NAMES[ex['flipper_model']]} [{flipper_family}]")
        print(f"     Stays: {MODEL_NAMES[ex['non_flipper_model']]} [{non_flipper_family}]")
        print(f"     Length: {ex['text_length']:,} chars")

    print("\n📌 DYNAMIC EXAMPLES:")
    for i, ex in enumerate(selected_dynamic, 1):
        flipper_family = get_model_family(ex["flipper_model"])
        non_flipper_family = get_model_family(ex["non_flipper_model"])
        print(f"\n  {i}. {FOLLOWUP_NAMES[ex['followup_type']]}")
        print(f"     Question ID: {ex['question_id']}")
        print(f"     Flips: {MODEL_NAMES[ex['flipper_model']]} [{flipper_family}]")
        print(f"     Stays: {MODEL_NAMES[ex['non_flipper_model']]} [{non_flipper_family}]")
        print(f"     Length: {ex['text_length']:,} chars")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
