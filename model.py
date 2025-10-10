from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from tqdm import tqdm


class LitellmModel:
    def __init__(
        self,
        model_id: str,
        parse_generation_fn=None,
        max_send_messages: int = 25,
        generate_kwargs: dict[str, Any] | None = None,
    ):
        import litellm
        from litellm import batch_completion

        # useful e.g. gpt5 not supporting temp
        litellm.drop_params = True
        self._batch_completion = batch_completion
        self.model_id = model_id
        self.parse_generation_fn = parse_generation_fn or (
            lambda responses: [response["choices"][0]["message"]["content"] for response in responses]
        )
        self.max_send_messages = max_send_messages
        self.generate_kwargs = generate_kwargs or {"temperature": 0.0}

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
        prepared = [self._format_message(p, system_prompt) for p in list(prompts)]
        outputs: List[Any] = []
        for start in tqdm(
            range(0, len(prepared), self.max_send_messages),
            desc=f"Processing batched requests (max bs={self.max_send_messages})",
        ):
            batch = prepared[start : start + self.max_send_messages]
            responses = self._batch_completion(
                model=self.model_id,
                messages=batch,
                **self.generate_kwargs,
            )
            outputs.extend(responses)
        return self.parse_generation_fn(outputs)


@dataclass
class LiteLLMConfig:
    id: str
    temperature: float
    max_tokens: int
    max_send_messages: int
    extra_kwargs: dict[str, Any]


@dataclass
class VLLMConfig:
    id: str
    temperature: float
    max_tokens: int
    tensor_parallel_size: int | None
    sampling_kwargs: dict[str, Any]


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
        self.engine = LLM(model=cfg.id, tensor_parallel_size=cfg.tensor_parallel_size)

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
    generate_kwargs = {"temperature": model_cfg.temperature, "max_tokens": model_cfg.max_tokens}
    generate_kwargs.update(extra)
    return LiteLLMConfig(
        id=model_cfg.id,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        max_send_messages=getattr(model_cfg, "max_send_messages", 25),
        extra_kwargs=generate_kwargs,
    )


def build_vllm_config(model_cfg) -> VLLMConfig:
    return VLLMConfig(
        id=model_cfg.id,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        tensor_parallel_size=getattr(model_cfg, "tensor_parallel_size", None),
        sampling_kwargs=dict(getattr(model_cfg, "sampling_kwargs", {}) or {}),
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
        )
        return ChatModel(impl)
    if model_type == "vllm":
        cfg = build_vllm_config(model_cfg)
        impl = VLLMModel(cfg)
        return ChatModel(impl)
    raise ValueError(f"Unknown model type '{model_type}'")
