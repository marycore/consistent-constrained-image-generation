from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.closed_api.config import load_generation_config
from src.closed_api.pipeline import run_closed_generation
from src.closed_api.registry import build_closed_registry


class ClosedApiPipelineTests(unittest.TestCase):
    def test_registry_contains_all_target_models(self):
        reg = build_closed_registry()
        expected = {
            "openai_gpt_image_2",
            "openai_gpt_image_1_5",
            "openai_dalle_3",
            "google_nano_banana",
            "google_imagen_4",
            "ideogram_3",
            "recraft",
            "firefly",
            "nova_canvas",
            "seedream",
            "grok_imagine",
        }
        self.assertEqual(expected, set(reg.keys()))

    def test_config_parsing_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text("models:\n  closed: [openai_dalle_3]\n", encoding="utf-8")
            cfg = load_generation_config(cfg_path)
            self.assertEqual(cfg["generation"]["timeout_seconds"], 180)
            self.assertEqual(cfg["budget"]["track_estimated_cost"], True)

    def test_dry_run_writes_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                json.dumps({"id": "p1", "complexity_level": "L0", "prompt": "test"}) + "\n",
                encoding="utf-8",
            )
            cfg = root / "cfg.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "models:",
                        "  closed: [openai_dalle_3]",
                        "prompts:",
                        f"  path: {prompts}",
                    ]
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                stats = run_closed_generation(config_path=str(cfg), dry_run=True)
                self.assertEqual(stats["attempted"], 1)
                meta = root / "outputs" / "generation_metadata.jsonl"
                self.assertTrue(meta.exists())
            finally:
                os.chdir(cwd)

    def test_resume_skips_existing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                json.dumps({"id": "p1", "complexity_level": "L0", "prompt": "test"}) + "\n",
                encoding="utf-8",
            )
            cfg = root / "cfg.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "models:",
                        "  closed: [openai_dalle_3]",
                        "prompts:",
                        f"  path: {prompts}",
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "outputs" / "generations" / "openai_dalle_3" / "L0" / "p1"
            out.mkdir(parents=True, exist_ok=True)
            (out / "seed_0_img_0.png").write_bytes(b"fake")
            cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                stats = run_closed_generation(config_path=str(cfg), dry_run=False, resume=True)
                self.assertEqual(stats["skipped"], 1)
            finally:
                os.chdir(cwd)

    def test_missing_api_key_fails_cleanly(self):
        reg = build_closed_registry()
        model = reg["openai_dalle_3"]
        # Force missing key
        import os

        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            res = model.generate(
                prompt="x",
                output_path="/tmp/does_not_matter.png",
                prompt_id="p1",
                complexity_level="L0",
            )
            self.assertFalse(res.success)
            self.assertIn("OPENAI_API_KEY", res.error_message or "")
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
