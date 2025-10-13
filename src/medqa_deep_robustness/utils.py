from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd
import yaml

ANSWER_SUFFIX = (
    'Response (think step by step and then end with "Final Answer:" '
    "followed by *only* the letter corresponding to the correct answer enclosed in parentheses)"
)

INSTRUCTION_PREFIX = (
    "Instructions: The following are multiple choice questions about medical knowledge. "
    "Solve them in a step-by-step fashion, starting by summarizing the available information. "
    "Output a single option from the four options as the final answer."
)

FINAL_ANSWER_REGEX = re.compile(r"final answer\s*:\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE)
FALLBACK_ANSWER_REGEX = re.compile(
    r"(?:(?:the\s+)?final\s+answer\s+is\s+\$?\\boxed\{([A-D])\}\$?|"
    r"final answer\s*:\s*(?:the\s+final\s+answer\s+is\s+)?\$?\\boxed\{([A-D])\}\$?|"
    r"(?:the\s+)?final\s+answer\s+is\s+\(?([A-D])\)?|"
    r"final answer\s*:\s*\(?\s*([A-D])\s*\)?|"
    r"\*{1,2}final\s+answer:?\*{1,2}\s*\(?([A-D])\)?)",
    re.IGNORECASE,
)


@dataclass
class FollowupTemplate:
    name: str
    kind: str
    template: Optional[str] = None
    column: Optional[str] = None
    prefix: str = ""
    join_with: str = " "
    allow_random_letter: bool = False
    dataset_columns: tuple[str, ...] = ()

    def render(self, row: Dict[str, Any], prev_pred: Optional[str]) -> str:
        if self.kind == "static":
            if self.template is None:
                raise ValueError(f"Followup '{self.name}' requires a template")
            message = self.template
        elif self.kind == "column":
            if self.column is None:
                raise ValueError(f"Followup '{self.name}' requires a column")
            if self.column not in row:
                raise KeyError(f"Column '{self.column}' not present in row for followup '{self.name}'")
            pieces = [self.prefix.strip(), str(row[self.column]).strip()]
            message = self.join_with.join(part for part in pieces if part)
        else:
            raise ValueError(f"Unsupported followup kind '{self.kind}'")

        placeholders: Dict[str, Any] = {
            "prev_letter": (prev_pred or "").upper() if prev_pred else "",
            "gold_letter": str(row.get("answer_idx", "")).strip().upper()[:1],
        }

        for col in self.dataset_columns:
            if col not in row:
                raise KeyError(f"Column '{col}' not present in row for followup '{self.name}'")
            placeholders[col] = str(row[col]).strip()

        if self.allow_random_letter:
            options = row.get("options", {})
            gold = placeholders["gold_letter"]
            wrong_letters = [opt for opt in options if opt != gold]
            if not wrong_letters:
                raise ValueError("No alternate answer choices to reference in followup template.")
            choice = random.choice(wrong_letters)
            placeholders.update(
                {
                    "incorr_letter": choice,
                    "incorr_text": options.get(choice, "another option"),
                }
            )
        try:
            message = message.format(**placeholders)
        except KeyError:
            pass
        return append_answer_suffix(message)


def append_answer_suffix(message: str, suffix: str = ANSWER_SUFFIX) -> str:
    if not suffix:
        return message.strip()
    spacer = " " + suffix.strip() if suffix and not suffix.endswith(" ") else suffix
    base = message.strip()
    return (base + spacer).strip() if base else suffix.strip()


def format_medqa_prompt(
    row: Dict[str, Any],
    instruction_prefix: str = INSTRUCTION_PREFIX,
    response_suffix: str = ANSWER_SUFFIX,
) -> str:
    question = str(row.get("question", "")).strip()
    options = row.get("options", {})
    options_text = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
    prompt = f"{instruction_prefix.strip()}\n\nQuestion: {question}\n{options_text}\n\n{response_suffix.strip()}"
    return prompt


def evaluate_generation(gold_letter: str, generation: str) -> Dict[str, Any]:
    if not generation:
        return {"gold": gold_letter.upper(), "pred": None, "correct": False, "raw": generation}

    match = FINAL_ANSWER_REGEX.search(generation)
    if match:
        pred = match.group(1).upper()
        return {
            "gold": gold_letter.upper(),
            "pred": pred,
            "correct": pred == gold_letter.upper(),
            "raw": generation,
        }

    matches = list(FALLBACK_ANSWER_REGEX.finditer(generation))
    if matches:
        last_match = matches[-1]
        pred = None
        for idx in range(1, 6):
            candidate = last_match.group(idx)
            if candidate:
                pred = candidate.upper()
                break
        return {
            "gold": gold_letter.upper(),
            "pred": pred,
            "correct": pred == gold_letter.upper(),
            "raw": generation,
        }

    return {"gold": gold_letter.upper(), "pred": None, "correct": False, "raw": generation}


def compute_flips(base_details: List[Dict[str, Any]], follow_details: List[Dict[str, Any]]):
    base_df = pd.DataFrame(base_details)[["id", "pred", "correct", "response"]].rename(
        columns={"pred": "pred_base", "correct": "correct_base", "response": "resp_base"}
    )
    follow_df = pd.DataFrame(follow_details)[["id", "pred", "correct", "response"]].rename(
        columns={"pred": "pred_follow", "correct": "correct_follow", "response": "resp_follow"}
    )
    merged = base_df.merge(follow_df, on="id", how="inner")
    flip_ci = ((merged.correct_base) & (~merged.correct_follow)).sum()
    flip_ic = ((~merged.correct_base) & (merged.correct_follow)).sum()
    stay_c = ((merged.correct_base) & (merged.correct_follow)).sum()
    stay_i = ((~merged.correct_base) & (~merged.correct_follow)).sum()
    changed = (merged.pred_base != merged.pred_follow).sum()
    rate = (changed / len(merged)) if len(merged) else 0.0
    return {
        "flip_CI": int(flip_ci),
        "flip_IC": int(flip_ic),
        "stay_C": int(stay_c),
        "stay_I": int(stay_i),
        "changed": int(changed),
        "flip_rate": rate,
    }


def load_followup_templates(path: Union[Path, str]) -> Dict[str, FollowupTemplate]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    entries = payload.get("followups", {}) if isinstance(payload, dict) else {}
    templates: Dict[str, FollowupTemplate] = {}
    for name, raw in entries.items():
        if isinstance(raw, str):
            templates[name] = FollowupTemplate(name=name, kind="static", template=raw)
            continue
        kind = raw.get("kind", "static")
        if kind == "column":
            templates[name] = FollowupTemplate(
                name=name,
                kind="column",
                column=raw["column"],
                prefix=raw.get("prefix", ""),
                join_with=raw.get("join_with", " "),
                dataset_columns=tuple(raw.get("dataset_columns", [])),
            )
        elif kind == "static":
            templates[name] = FollowupTemplate(
                name=name,
                kind="static",
                template=raw["template"],
                allow_random_letter=raw.get("allow_random_letter", False),
                dataset_columns=tuple(raw.get("dataset_columns", [])),
            )
        else:
            raise ValueError(f"Unsupported followup kind '{kind}' for '{name}'")
    return templates


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def stable_config_signature(cfg_dict: Dict[str, Any]) -> str:
    blob = json.dumps(cfg_dict, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


def seed_everything(seed: int) -> None:
    random.seed(seed)


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))
    sep = "-+-".join("-" * width for width in widths)
    body = [" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in str_rows]
    return "\n".join([header_line, sep, *body]) if rows else header_line
