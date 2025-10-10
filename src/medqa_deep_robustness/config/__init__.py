from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.conf import HydraConf, JobConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from ..utils import ANSWER_SUFFIX, INSTRUCTION_PREFIX


@dataclass
class PromptConfig:
    instruction_prefix: str = INSTRUCTION_PREFIX
    response_suffix: str = ANSWER_SUFFIX
    system_prompt: Optional[str] = None


@dataclass
class RunConfig:
    dataset_name: str = "dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust"
    split: str = "train"
    n_rows: Any = "all"
    followups_file: str = "followups.yaml"
    followups: Any = field(default_factory=lambda: [
        "br_authority_prior",
        "br_autograder_prior",
        "br_commitment_alignment",
        "br_recency_prior",
        "br_social_proof_prior",
        "context_rag_style",
        "alternative_context",
        "edge_case_context",
    ])
    results_dir: str = "results"
    overwrite: bool = False
    seed: int = 0


@dataclass
class BaseModelConfig:
    id: str = "openai/gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    type: str = "litellm"


@dataclass
class LiteLLMConfig(BaseModelConfig):
    type: str = "litellm"
    max_send_messages: int = 100
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLLMConfig(BaseModelConfig):
    type: str = "vllm"
    tensor_parallel_size: Optional[int] = 1
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_model_len: int = 16384


@dataclass
class AppConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"model": "litellm"},
        "_self_",
    ])
    model: BaseModelConfig = field(default=MISSING)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    run: RunConfig = field(default_factory=RunConfig)
    hydra: HydraConf = field(default_factory=lambda: HydraConf(job=JobConf(chdir=False)))


cs = ConfigStore.instance()
cs.store(name="app", node=AppConfig)
cs.store(group="model", name="litellm", node=LiteLLMConfig)
cs.store(group="model", name="vllm", node=VLLMConfig)

__all__ = [
    "PromptConfig",
    "RunConfig",
    "BaseModelConfig",
    "LiteLLMConfig",
    "VLLMConfig",
    "AppConfig",
]
