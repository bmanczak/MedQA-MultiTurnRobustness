from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from .config import AppConfig  # noqa: F401 - register config store
from .model import build_model
from .utils import (
    compute_flips,
    evaluate_generation,
    format_medqa_prompt,
    format_table,
    load_followup_templates,
    read_jsonl,
    seed_everything,
    stable_config_signature,
    write_jsonl,
)


def _sanitize(text: str) -> str:
    return text.replace("/", "_").replace(":", "_")


def _load_dataset(cfg: DictConfig):
    split = cfg.run.split
    if cfg.run.n_rows is not None and cfg.run.n_rows != "all":
        split = f"{split}[:{int(cfg.run.n_rows)}]"
    ds = load_dataset(cfg.run.dataset_name, split=split)
    return [ds[idx] for idx in range(len(ds))]


def _prepare_run_dir(cfg: DictConfig, root: Path, n_examples: int) -> Path:
    followups_value = cfg.run.followups
    if followups_value != "all":
        followups_value = list(OmegaConf.to_container(followups_value, resolve=True))

    signature_payload: Dict[str, object] = {
        "model": cfg.model.id,
        "dataset": cfg.run.dataset_name,
        "split": cfg.run.split,
        "n_rows": n_examples,
        "system_prompt": cfg.prompts.system_prompt,
        "followups": followups_value,
    }
    # Include extra_kwargs in signature only when non-empty (backwards compatible)
    extra_kwargs = cfg.model.get("extra_kwargs", {})
    if extra_kwargs:
        signature_payload["extra_kwargs"] = dict(OmegaConf.to_container(extra_kwargs, resolve=True))
    signature = stable_config_signature(signature_payload)
    model_tag = _sanitize(cfg.model.id)
    dataset_tag = _sanitize(cfg.run.dataset_name)
    run_dir = root / cfg.run.results_dir / f"{model_tag}__{dataset_tag}__{cfg.run.split}__{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "followups").mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    return run_dir


def _load_followup_templates(root: Path, filename: str):
    candidate = root / filename
    if candidate.exists():
        return load_followup_templates(candidate)

    resource = resources.files("medqa_deep_robustness").joinpath("data", filename)
    try:
        with resources.as_file(resource) as packaged_path:
            return load_followup_templates(Path(packaged_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Followups file '{filename}' not found in working directory or package data") from exc


def _backfill_empty_base(
    rows: List[Dict],
    empty_ids: List[int],
    base_cache: Dict[int, Dict],
    cfg: DictConfig,
    system_prompt: str,
    model,
) -> None:
    """Try to re-generate base responses for rows with empty outputs.

    Runs up to two small-batch attempts and updates base_cache in-place.
    The caller is responsible for validating that no empties remain.
    """
    if not empty_ids:
        return
    max_attempts = 2
    for _ in range(max_attempts):
        pending = [i for i in empty_ids if str(base_cache.get(i, {}).get("response", "")).strip() == ""]
        if not pending:
            return
        prompts = [
            format_medqa_prompt(rows[idx], cfg.prompts.instruction_prefix, cfg.prompts.response_suffix)
            for idx in pending
        ]
        generations = model.generate(prompts, system_prompt=system_prompt)
        for idx, prompt, generation in zip(pending, prompts, generations):
            row = rows[idx]
            gold = str(row.get("answer_idx", "")).strip().upper()[:1]
            evaluation = evaluate_generation(gold, generation)
            base_cache[idx] = {
                "id": idx,
                "prompt": prompt,
                "response": generation,
                "gold": evaluation["gold"],
                "pred": evaluation["pred"],
                "correct": bool(evaluation["correct"]),
            }


def run_experiment(cfg: DictConfig) -> Path:
    try:
        root = Path(get_original_cwd())
    except ValueError:
        root = Path.cwd()
    seed_everything(int(cfg.run.seed))

    rows = _load_dataset(cfg)
    if not rows:
        raise ValueError("Dataset selection returned 0 rows")

    run_dir = _prepare_run_dir(cfg, root, len(rows))
    base_file = run_dir / "first_turn.jsonl"
    follow_dir = run_dir / "followups"

    followup_templates = _load_followup_templates(root, cfg.run.followups_file)
    follow_ups_to_execute = (
        list(followup_templates.keys())
        if cfg.run.followups == "all"
        else list(OmegaConf.to_container(cfg.run.followups, resolve=True))
    )
    missing = [name for name in follow_ups_to_execute if name not in followup_templates]
    if missing:
        raise ValueError(f"Missing followup definitions for: {missing}")

    model = None
    system_prompt = cfg.prompts.system_prompt or ""

    base_existing: List[Dict] = [] if cfg.run.overwrite else read_jsonl(base_file)
    base_cache = {
        rec["id"]: rec
        for rec in base_existing
        if isinstance(rec.get("id"), int) and 0 <= rec["id"] < len(rows) and str(rec.get("response", "")).strip() != ""
    }
    base_missing_ids = (
        list(range(len(rows))) if cfg.run.overwrite else [i for i in range(len(rows)) if i not in base_cache]
    )

    if base_missing_ids:
        if model is None:
            model = build_model(cfg.model)
        prompts = [
            format_medqa_prompt(rows[idx], cfg.prompts.instruction_prefix, cfg.prompts.response_suffix)
            for idx in base_missing_ids
        ]
        generations = model.generate(prompts, system_prompt=system_prompt)
        for idx, prompt, generation in zip(base_missing_ids, prompts, generations):
            row = rows[idx]
            gold = str(row.get("answer_idx", "")).strip().upper()[:1]
            evaluation = evaluate_generation(gold, generation)
            base_cache[idx] = {
                "id": idx,
                "prompt": prompt,
                "response": generation,
                "gold": evaluation["gold"],
                "pred": evaluation["pred"],
                "correct": bool(evaluation["correct"]),
            }

    # Fail-fast backfill: ensure no empty base responses before followups
    empty_base_ids = [i for i in range(len(rows)) if str(base_cache.get(i, {}).get("response", "")).strip() == ""]
    if empty_base_ids:
        if model is None:
            model = build_model(cfg.model)
        _backfill_empty_base(rows, empty_base_ids, base_cache, cfg, system_prompt, model)
        empty_base_ids = [i for i in range(len(rows)) if str(base_cache.get(i, {}).get("response", "")).strip() == ""]
        if empty_base_ids:
            raise RuntimeError(
                f"Base generation failed for rows {empty_base_ids} after backfill; aborting before followups."
            )

    try:
        base_records = [base_cache[i] for i in range(len(rows))]
    except KeyError as missing_idx:
        raise RuntimeError(f"Missing cached baseline result for row {missing_idx}") from missing_idx

    if cfg.run.overwrite or base_missing_ids or len(base_existing) != len(base_records):
        write_jsonl(base_file, base_records)
    elif not base_missing_ids:
        print(f"Loaded cached first-turn results from {base_file}")

    followup_outcomes: Dict[str, List[Dict]] = {}

    pending_prompts: List[List[str]] = []
    pending_meta: List[Dict[str, Any]] = []
    followup_caches: Dict[str, Dict[int, Dict]] = {}
    updated_followups: set[str] = set()

    for name in follow_ups_to_execute:
        template = followup_templates[name]
        outfile = follow_dir / f"{name}.jsonl"
        existing_records: List[Dict] = [] if cfg.run.overwrite else read_jsonl(outfile)
        cache = {
            rec["id"]: rec
            for rec in existing_records
            if isinstance(rec.get("id"), int)
            and 0 <= rec["id"] < len(rows)
            and str(rec.get("response", "")).strip() != ""
        }
        followup_caches[name] = cache
        missing_ids = list(range(len(rows))) if cfg.run.overwrite else [i for i in range(len(rows)) if i not in cache]

        if missing_ids:
            updated_followups.add(name)
            for idx in missing_ids:
                base_rec = base_records[idx]
                follow_prompt = template.render(rows[idx], base_rec.get("pred"))
                conversation = [base_rec["prompt"], base_rec["response"], follow_prompt]
                pending_prompts.append(conversation)
                pending_meta.append(
                    {
                        "name": name,
                        "id": idx,
                        "follow_prompt": follow_prompt,
                        "base_rec": base_rec,
                    }
                )
        else:
            final_records = [cache[i] for i in range(len(rows))]
            if cfg.run.overwrite or len(existing_records) != len(final_records):
                write_jsonl(outfile, final_records)
            followup_outcomes[name] = final_records

    if pending_prompts:
        if model is None:
            model = build_model(cfg.model)
        generations = model.generate(pending_prompts, system_prompt=system_prompt)
        for meta, generation in zip(pending_meta, generations):
            cache = followup_caches[meta["name"]]
            base_rec = meta["base_rec"]
            evaluation = evaluate_generation(base_rec["gold"], generation)
            cache[meta["id"]] = {
                "id": meta["id"],
                "followup": meta["name"],
                "prompt": meta["follow_prompt"],
                "response": generation,
                "gold": evaluation["gold"],
                "pred": evaluation["pred"],
                "correct": bool(evaluation["correct"]),
            }

        for name in updated_followups:
            cache = followup_caches[name]
            # Ensure alignment: any ids not produced get explicit empty records
            for i in range(len(rows)):
                if i not in cache:
                    base_rec = base_records[i]
                    cache[i] = {
                        "id": i,
                        "followup": name,
                        "prompt": "",
                        "response": "",
                        "gold": base_rec.get("gold", ""),
                        "pred": "",
                        "correct": False,
                    }
            final_records = [cache[i] for i in range(len(rows))]
            outfile = follow_dir / f"{name}.jsonl"
            write_jsonl(outfile, final_records)
            followup_outcomes[name] = final_records

    _print_summary(base_records, followup_outcomes)
    return run_dir


def _print_summary(base_records: List[Dict], followup_outcomes: Dict[str, List[Dict]]) -> None:
    base_n = len(base_records)
    base_correct = sum(int(rec.get("correct", False)) for rec in base_records)
    base_acc = base_correct / base_n if base_n else 0.0

    headers = ["Followup", "n", "Accuracy", "Δ vs Base", "Flip C→I", "Flip I→C", "Flip Rate"]
    rows: List[List[str]] = [
        [
            "first_turn",
            str(base_n),
            f"{base_acc*100:.1f}%",
            "-",
            "-",
            "-",
            "-",
        ]
    ]

    for name, records in followup_outcomes.items():
        # Exclude API failures (empty responses) from statistics and n
        filtered_records = [rec for rec in records if str(rec.get("response", "")).strip() != ""]
        n = len(filtered_records)
        correct = sum(int(rec.get("correct", False)) for rec in filtered_records)
        acc = correct / n if n else 0.0
        flips = compute_flips(base_records, filtered_records)
        rows.append(
            [
                name,
                str(n),
                f"{acc*100:.1f}%",
                f"{(acc - base_acc)*100:+.1f}%",
                str(flips["flip_CI"]),
                str(flips["flip_IC"]),
                f"{flips["flip_rate"]*100:.1f}%",
            ]
        )

    table = format_table(headers, rows)
    print("\nEvaluation Summary:\n" + table)
