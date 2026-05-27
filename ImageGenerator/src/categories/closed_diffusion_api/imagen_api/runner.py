from __future__ import annotations

from dataclasses import dataclass

from ....common.closed_bridge import run_closed_via_registry
from ....common.registry import register
from ....common.types import Runner, Category, PromptRecord, Mode, FinetuneConfig


@dataclass
class ImagenAPIRunner(Runner):
    model_id: str = "imagen_api"
    category: Category = "closed_diffusion_api"

    def run(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
    ) -> None:
        run_closed_via_registry(
            legacy_model_id=self.model_id,
            provider_model_id="google_imagen_4",
            category=self.category,
            prompts=prompts,
            mode=mode,
            seed=seed,
            output_root=output_root,
        )

    def finetune(
        self,
        *,
        dataset_path: str,
        images_root: str,
        out_dir: str,
        config: FinetuneConfig,
        init_ckpt_dir: str | None = None,
    ) -> None:
        raise NotImplementedError("Fine-tuning is not supported for imagen_api (closed API).")

    def run_finetuned(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
        ckpt_dir: str,
    ) -> None:
        raise NotImplementedError("run_finetuned is not supported for imagen_api (closed API).")


runner = ImagenAPIRunner()
register(runner)

