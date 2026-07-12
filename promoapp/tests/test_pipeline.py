import tempfile
import unittest
from pathlib import Path

from promoapp.pipeline import ingest, scenes
from promoapp.pipeline.run import PipelineConfig, run_pipeline

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_input.mp4"


class IngestTests(unittest.TestCase):
    def test_probe_reads_fixture_metadata(self):
        info = ingest.probe(FIXTURE)
        self.assertGreater(info.duration, 5.0)
        self.assertLess(info.duration, 7.0)
        self.assertTrue(info.has_audio)
        self.assertEqual((info.width, info.height), (320, 180))

    def test_probe_rejects_wrong_extension(self):
        with self.assertRaises(ingest.IngestError):
            ingest.probe(FIXTURE.with_suffix(".txt"))

    def test_validate_rejects_too_long(self):
        info = ingest.probe(FIXTURE)
        with self.assertRaises(ingest.IngestError):
            ingest.validate(info, file_size_bytes=1000, max_mb=500, max_minutes=0.01)


class SceneDetectionTests(unittest.TestCase):
    def test_detects_at_least_three_scenes(self):
        info = ingest.probe(FIXTURE)
        scene_list = scenes.detect_scenes(FIXTURE, info.duration, threshold=15.0)
        self.assertGreaterEqual(len(scene_list), 3)


class MvpPipelineTests(unittest.TestCase):
    def test_run_pipeline_mvp_produces_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "promo.mp4"
            config = PipelineConfig.for_profile("mvp", target_duration=15, scene_threshold=15.0)

            result = run_pipeline(FIXTURE, output_path, config)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertLessEqual(result.output_duration_s, 15.5)
            self.assertGreaterEqual(len(result.scene_scores), 3)
            self.assertTrue(any(s.selected for s in result.scene_scores))
