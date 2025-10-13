#!/usr/bin/env python3
"""
Tiny Together API smoke test (no third-party deps).

Purpose:
  - Validate that Together model IDs respond via the OpenAI-compatible API.
  - Run two minimal prompts against two large models to confirm routing.

Usage:
  - Ensure TOGETHER_API_KEY is set in the environment.
  - python scripts/together_smoke.py

Raises:
  - AssertionError: If required environment variables are missing or responses are empty.
  - urllib.error.HTTPError/URLError: On HTTP errors.
"""
from __future__ import annotations

import json
import os
import ssl
import sys
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TOGETHER_BASE_URL = "https://api.together.xyz"
MODEL_IDS: List[str] = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen2.5-72B-Instruct",
]


def get_api_key() -> str:
    """Return Together API key from env.

    Inputs: none
    Outputs: API key string
    Exceptions: AssertionError if TOGETHER_API_KEY is missing
    """
    api_key = os.environ.get("TOGETHER_API_KEY", "").strip() or os.environ.get("TOGETHERAI_API_KEY", "").strip()
    assert api_key, "TOGETHER_API_KEY must be set"
    return api_key


def build_messages(prompt: str) -> List[Dict[str, str]]:
    """Build a minimal chat message array.

    Inputs: prompt (user content)
    Outputs: List of role/content messages
    Exceptions: AssertionError if prompt is empty
    """
    assert isinstance(prompt, str) and prompt.strip(), "prompt must be a non-empty string"
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]


def post_chat_completion(model_id: str, api_key: str) -> str:
    """Post a minimal chat.completions request and return the first message content.

    Inputs:
      - model_id: Together model identifier
      - api_key: Together API key
    Outputs:
      - First choice message content string
    Exceptions:
      - AssertionError on invalid inputs or empty content
      - HTTPError/URLError on network/API issues
    """
    assert model_id.strip(), "model_id must be a non-empty string"
    assert api_key.strip(), "api_key must be a non-empty string"

    payload = {
        "model": model_id,
        "messages": build_messages("Reply with: OK"),
        "temperature": 0.2,
        "max_tokens": 32,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{TOGETHER_BASE_URL}/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    # Create default SSL context to avoid env-specific SSL issues.
    ctx = ssl.create_default_context()
    with urlopen(req, context=ctx, timeout=60) as resp:
        assert resp.status == 200, f"HTTP status {resp.status}"
        body = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(body)
    # Validate structure
    choices = obj.get("choices") or []
    assert isinstance(choices, list) and choices, "Missing choices in response"
    message = (choices[0].get("message") or {}).get("content", "").strip()
    assert message, "Empty response message"
    return message


def list_models(api_key: str) -> List[str]:
    """Fetch Together /v1/models and return list of model IDs.

    Inputs:
      - api_key: Together API key
    Outputs:
      - List of model id strings
    Exceptions:
      - HTTPError/URLError on network/API issues
      - AssertionError on unexpected response structure
    """
    req = Request(
        f"{TOGETHER_BASE_URL}/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    ctx = ssl.create_default_context()
    with urlopen(req, context=ctx, timeout=60) as resp:
        assert resp.status == 200, f"HTTP status {resp.status}"
        body = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(body)
    data = obj.get("data") or []
    assert isinstance(data, list), "models response malformed"
    ids: List[str] = []
    for item in data:
        mid = (item.get("id") or "").strip()
        if mid:
            ids.append(mid)
    return ids


def main() -> None:
    """Run smoke tests for two Together models and print their outputs."""
    api_key = get_api_key()
    try:
        ids = list_models(api_key)
        has_gpt_oss_120b = any(mid.endswith("gpt-oss-120b") or mid == "openai/gpt-oss-120b" for mid in ids)
        has_qwen_72b = any("Qwen2.5-72B-Instruct" in mid for mid in ids)
        print(f"[models] contains gpt-oss-120b: {has_gpt_oss_120b}; contains Qwen2.5-72B-Instruct: {has_qwen_72b}")
    except (AssertionError, HTTPError, URLError) as e:
        print(f"[models] ERROR: {e}", file=sys.stderr)
    for model_id in MODEL_IDS:
        try:
            text = post_chat_completion(model_id, api_key)
            print(f"[{model_id}] => {text}")
        except (AssertionError, HTTPError, URLError) as e:
            print(f"[{model_id}] ERROR: {e}", file=sys.stderr)
            # continue to next model without aborting


if __name__ == "__main__":
    main()
