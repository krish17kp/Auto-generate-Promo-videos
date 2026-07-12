import tempfile
import unittest
from pathlib import Path

from promoapp.pipeline.run import PipelineConfig, run_pipeline_multi_aspect

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "transcript_demo.mp4"


class CaptionRenderTests(unittest.TestCase):
    """M3.2: burned-in captions render without error across all three aspect presets."""

    def test_captions_render_on_all_aspect_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            aspects = ["16:9", "9:16", "1:1"]
            output_paths = {a: Path(tmp) / f"captioned_{a.replace(':', 'x')}.mp4" for a in aspects}
            config = PipelineConfig.for_profile("advanced", target_duration=15, scene_threshold=15.0)

            results = run_pipeline_multi_aspect(FIXTURE, output_paths, config)

            self.assertEqual(set(results), set(aspects))
            for aspect, result in results.items():
                with self.subTest(aspect=aspect):
                    self.assertTrue(Path(result.output_path).exists())
                    self.assertGreater(Path(result.output_path).stat().st_size, 0)
                    self.assertGreater(result.output_duration_s, 0)
