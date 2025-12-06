from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from litellm import batch_completion
from tqdm import tqdm

REFUSAL_MARKER = "The model has not returned the response"


def _safe_get(obj: Any, key: str) -> Any:
    """Retrieve `key` from dict-like or attribute-bearing objects."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _stringify_content(content: Any) -> str | None:
    """Normalize provider content payloads into plain strings."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
        return "".join(parts) if parts else None
    return str(content)


def _default_parse_generation(responses) -> List[str | None]:
    """Convert LiteLLM responses into strings while tagging refusals."""
    parsed: List[str | None] = []
    for response in responses:
        choices = _safe_get(response, "choices") or []
        first_choice = choices[0] if choices else None
        finish_reason = (_safe_get(first_choice, "finish_reason") or "").lower()
        message = _safe_get(first_choice, "message")
        raw_content = _safe_get(message, "content")
        content = _stringify_content(raw_content)
        if content is None and finish_reason == "refusal":
            parsed.append(REFUSAL_MARKER)
            continue
        parsed.append(content)
    return parsed


class LitellmModel:
    def __init__(
        self,
        model_id: str,
        parse_generation_fn=None,
        max_send_messages: int = 25,
        generate_kwargs: dict[str, Any] | None = None,
        max_retries: int = 5,
    ):
        self._batch_completion = batch_completion
        self.model_id = model_id
        self.parse_generation_fn = parse_generation_fn or _default_parse_generation
        self.max_send_messages = max_send_messages
        self.generate_kwargs = generate_kwargs or {"temperature": 0.0}
        self.max_retries = max(0, max_retries)

    @staticmethod
    def _format_message(message: str | Sequence[str], system_prompt: str = ""):
        if isinstance(message, str):
            message = [message]
        convo = []
        if system_prompt:
            convo.append({"role": "system", "content": system_prompt})
        for idx, part in enumerate(message):
            role = "user" if idx % 2 == 0 else "assistant"
            convo.append({"role": role, "content": part})
        return convo

    def batch_call(self, prompts: Iterable[str | Sequence[str]], system_prompt: str = "") -> List[str]:
        """Batch-generate model outputs for prompts.

        Args:
            prompts: An iterable of prompts; each prompt is a string or an alternating user/assistant sequence.
            system_prompt: Optional system message prepended to each conversation.

        Returns:
            A list of generated strings, one per input prompt, in order.

        Raises:
            RuntimeError: Internally raised to trigger a retry when the provider returns empty text for any item.
        """
        prepared = [self._format_message(p, system_prompt) for p in list(prompts)]
        outputs: List[str] = []

        for start in tqdm(
            range(0, len(prepared), self.max_send_messages),
            desc=f"Processing batched requests (max bs={self.max_send_messages})",
        ):
            batch = prepared[start : start + self.max_send_messages]
            attempt = 0
            while True:
                try:
                    responses = self._batch_completion(
                        model=self.model_id,
                        messages=batch,
                        **self.generate_kwargs,
                    )
                    for resp in responses:
                        if isinstance(resp, Exception):
                            raise resp
                    parsed = self.parse_generation_fn(responses)
                    if not isinstance(parsed, list):
                        parsed = list(parsed)
                    if len(parsed) != len(batch):
                        raise RuntimeError("Unexpected number of responses from LiteLLM")
                    # Retry if any generation is empty/whitespace or not a string
                    if any((not isinstance(x, str)) or (not x.strip()) for x in parsed):
                        raise RuntimeError("Empty generation(s) in batch")
                    outputs.extend(parsed)
                    break
                except Exception as exc:  # noqa: BLE001
                    attempt += 1
                    if attempt > self.max_retries:
                        outputs.extend([""] * len(batch))
                        break
                    time.sleep(min(2 ** (attempt - 1) * 0.5, 5.0))

        return outputs


@dataclass
class LiteLLMConfig:
    id: str
    temperature: float
    max_tokens: int
    max_send_messages: int
    extra_kwargs: dict[str, Any]
    max_retries: int = 5


@dataclass
class VLLMConfig:
    id: str
    temperature: float
    max_tokens: int
    tensor_parallel_size: int | None
    sampling_kwargs: dict[str, Any]
    max_model_len: int


class VLLMModel:
    def __init__(self, cfg: VLLMConfig):
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install vllm and transformers to use model.type=vllm") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.id)
        sampling = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
        sampling.update(cfg.sampling_kwargs)
        self.sampling_params = SamplingParams(**sampling)
        self.engine = LLM(
            model=cfg.id,
            tensor_parallel_size=cfg.tensor_parallel_size,
            max_model_len=cfg.max_model_len,
        )

    def _format(self, prompt: str | Sequence[str], system_prompt: str) -> str:
        messages: List[dict[str, str]] = []
        if isinstance(prompt, str):
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for idx, chunk in enumerate(prompt):
                role = "user" if idx % 2 == 0 else "assistant"
                messages.append({"role": role, "content": chunk})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def batch_call(self, prompts: Iterable[str | Sequence[str]], system_prompt: str = "") -> List[str]:
        prompt_list = list(prompts)
        prompt_texts = [self._format(p, system_prompt) for p in prompt_list]
        outputs = self.engine.generate(prompt_texts, sampling_params=self.sampling_params)
        return [out.outputs[0].text for out in outputs]


def build_litellm_config(model_cfg) -> LiteLLMConfig:
    extra = dict(getattr(model_cfg, "extra_kwargs", {}) or {})

    # Normalize provider-specific parameters
    model_id = str(model_cfg.id)
    lowered = model_id.lower()
    is_gpt5 = "gpt-5" in lowered
    is_anthropic = lowered.startswith("anthropic/") or "claude" in lowered
    is_gemini = lowered.startswith("gemini/") or "gemini-" in lowered

    temperature = model_cfg.temperature
    max_tokens = model_cfg.max_tokens

    # GPT-5 requires temperature=1
    if is_gpt5:
        temperature = 1.0

    # Claude Sonnet: if extended thinking requested via extra_kwargs.reasoning_effort, enforce temp=1
    if is_anthropic and ("reasoning_effort" in extra):
        temperature = 1.0
        # Provide a small default thinking budget if not provided by user
        extra.setdefault("thinking", {"type": "enabled", "budget_tokens": 1024})
        # Ensure max_tokens exceeds thinking budget
        try:
            budget = int(extra["thinking"]["budget_tokens"])  # type: ignore[index]
            if max_tokens <= budget:
                max_tokens = budget + 256
        except Exception:  # noqa: BLE001
            pass

    # Gemini models: disable thinking mode for flash variants
    if is_gemini:
        if "gemini-2.5-flash" in lowered or "gemini-2.0-flash" in lowered:
            if "thinking" not in extra:
                extra["thinking"] = {"type": "disabled", "budget_tokens": 0}

    generate_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    generate_kwargs.update(extra)
    return LiteLLMConfig(
        id=model_cfg.id,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        max_send_messages=getattr(model_cfg, "max_send_messages", 25),
        extra_kwargs=generate_kwargs,
        max_retries=getattr(model_cfg, "max_retries", 5),
    )


def build_vllm_config(model_cfg) -> VLLMConfig:
    return VLLMConfig(
        id=model_cfg.id,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        tensor_parallel_size=getattr(model_cfg, "tensor_parallel_size", None),
        sampling_kwargs=dict(getattr(model_cfg, "sampling_kwargs", {}) or {}),
        max_model_len=getattr(model_cfg, "max_model_len", 16384),
    )


class ChatModel:
    def __init__(self, impl):
        self.impl = impl

    def generate(self, prompts: List[str | Sequence[str]], system_prompt: str = "") -> List[str]:
        return self.impl.batch_call(prompts, system_prompt=system_prompt)


def build_model(model_cfg) -> ChatModel:
    model_type = getattr(model_cfg, "type", "litellm")
    if model_type == "litellm":
        cfg = build_litellm_config(model_cfg)
        impl = LitellmModel(
            model_id=cfg.id,
            max_send_messages=cfg.max_send_messages,
            generate_kwargs=cfg.extra_kwargs,
            max_retries=cfg.max_retries,
        )
        return ChatModel(impl)
    if model_type == "vllm":
        cfg = build_vllm_config(model_cfg)
        impl = VLLMModel(cfg)
        return ChatModel(impl)
    raise ValueError(f"Unknown model type '{model_type}'")
