from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .runner import run_experiment


@hydra.main(version_base=None, config_name="app")
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
