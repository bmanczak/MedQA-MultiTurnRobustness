"""
Temporary verification script for LiteLLM provider/model integration.

This script performs live smoke tests against multiple providers (OpenAI, Anthropic,
and xAI) to validate that:

- The configured model identifiers are accepted by LiteLLM
- Provider API keys are detected from the environment
- Reasoning control works as expected for models that support it
  - GPT‑5 series: reasoning is controlled via the `reasoning_effort` parameter; to run
    with reasoning turned OFF, simply omit `reasoning_effort`.
    Note: GPT‑5 requires temperature=1.0 regardless of reasoning setting.
  - Anthropic Claude Sonnet 4.x: extended thinking is enabled when `reasoning_effort`
    is set (we pass Anthropic `thinking={type:"enabled", budget_tokens:1024}` and
    enforce temperature=1.0). To turn OFF reasoning, omit `reasoning_effort`; this will
    disable extended thinking and temperature can be < 1 (we use 0.0 by default here).
  - xAI Grok: we exercise `xai/grok-4-fast-non-reasoning` and `xai/grok-4-0709`.

Environment variables expected:
- OPENAI_API_KEY  — required for OpenAI-compatible (GPT) models
- ANTHROPIC_API_KEY — required for Anthropic Claude models
- XAI_API_KEY — required for xAI Grok models

What this script does:
1) Builds a test matrix including both reasoning ON and reasoning OFF for relevant models
2) Sends a minimal prompt via LiteLLM completion API
3) Prints a short snippet of the response (or the raw response if provider shape differs)

Usage:
  - Ensure dependencies are installed and API keys exported
  - Run: `python tmp_verify_models.py`

Notes:
- This script is a smoke test; it is not a benchmark. Keep test budgets small.
- For Anthropic extended thinking, we set budget_tokens=1024 to satisfy API constraints.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

# Import litellm dynamically to avoid static import linting issues
_litellm = importlib.import_module("litellm")
completion = getattr(_litellm, "completion")


def has_env(var_name: str) -> bool:
    """Return True if an environment variable exists and is non-empty."""
    return bool(os.environ.get(var_name, "").strip())


def call_model(
    model_id: str,
    prompt: str,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call a chat model via LiteLLM and return the raw response payload.

    Args:
        model_id: Provider/model identifier understood by LiteLLM (e.g., 'gpt-5-mini-2025-08-07', 'anthropic/claude-sonnet-4-5-20250929', 'xai/grok-4-0709').
        prompt: The user prompt to send.
        reasoning_effort: Optional reasoning effort for models that support it ('minimal'|'low'|'medium'|'high').

    Returns:
        The LiteLLM response dictionary, normalized to OpenAI-style shape.

    Raises:
        AssertionError: If the response shape does not include 'choices[0].message.content'.
    """
    messages = [{"role": "user", "content": prompt}]
    # Temperature rules:
    # - GPT-5 series only supports temperature=1.0
    # - Claude Sonnet requires temperature=1 when extended thinking is enabled (reasoning_effort provided)
    lowered = model_id.lower()
    is_gpt5 = "gpt-5" in lowered
    is_anthropic = lowered.startswith("anthropic/") or "claude" in lowered
    temp = 1.0 if is_gpt5 or (is_anthropic and reasoning_effort is not None) else 0.0

    kwargs: Dict[str, Any] = {"model": model_id, "messages": messages, "temperature": temp, "max_tokens": 1536}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
        if is_anthropic:
            # Extended thinking requires >=1024 budget tokens
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}

    resp = completion(**kwargs)
    return resp


def _provider_env_for(model_id: str) -> str:
    """Return the required API key env var for a given model identifier.

    Args:
        model_id: The LiteLLM model identifier.

    Returns:
        The environment variable name that must be set for the provider.

    Raises:
        AssertionError: If the model_id cannot be mapped to a known provider.
    """
    lowered = model_id.lower()
    if lowered.startswith("xai/") or "grok-" in lowered:
        return "XAI_API_KEY"
    if (
        lowered.startswith("anthropic/")
        or lowered.startswith("claude-")
        or lowered.startswith("claude_")
        or "claude" in lowered
    ):
        return "ANTHROPIC_API_KEY"
    # Default to OpenAI-compatible models (OpenAI/Azure OpenAI)
    return "OPENAI_API_KEY"


def main() -> None:
    """Run smoke checks against the listed models to verify configuration.

    This script will assert API keys exist and perform a minimal generation for each
    configured model. It is intended as a quick end-to-end verification.
    """
    tests: List[Dict[str, Any]] = [
        {"model": "openai/gpt-5-mini-2025-08-07", "reasoning_effort": "medium"},
        {"model": "openai/gpt-5-2025-08-07", "reasoning_effort": "high"},
        # GPT-5 with reasoning OFF: omit reasoning_effort
        {"model": "openai/gpt-5-2025-08-07", "reasoning_effort": None},
        {"model": "openai/gpt-4o-2024-08-06", "reasoning_effort": None},
        {"model": "anthropic/claude-sonnet-4-5-20250929", "reasoning_effort": "high"},
        {"model": "anthropic/claude-sonnet-4-20250514", "reasoning_effort": "medium"},
        # Anthropic with reasoning OFF: omit reasoning_effort (disables extended thinking)
        {"model": "anthropic/claude-sonnet-4-5-20250929", "reasoning_effort": None},
        {"model": "anthropic/claude-sonnet-4-20250514", "reasoning_effort": None},
        {"model": "xai/grok-4-fast-non-reasoning", "reasoning_effort": None},
        {"model": "xai/grok-4-0709", "reasoning_effort": None},
    ]

    prompt = "Return the single word: ok"
    for case in tests:
        env_var = _provider_env_for(case["model"])  # determine provider key
        if not has_env(env_var):
            print(f"SKIP {case['model']} (missing {env_var})")
            continue
        try:
            resp = call_model(case["model"], prompt, case["reasoning_effort"])  # may vary by provider
            text = None
            if isinstance(resp, dict) and "choices" in resp and resp.get("choices"):
                text = (resp["choices"][0].get("message", {}).get("content", "") or "").strip()
            snippet = text if text else str(resp)[:120]
            print(f"OK {case['model']}: {snippet}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {case['model']}: {exc}")

    print("All model smoke tests passed.")


if __name__ == "__main__":
    main()
