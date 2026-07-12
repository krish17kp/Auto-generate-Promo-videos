import shutil
import subprocess
import tempfile
from pathlib import Path

from django.test import TestCase, override_settings

from promoapp import jobs
from promoapp.models import PromoJob, VideoUpload

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "demo_input.mp4"


class MultiAspectJobTests(TestCase):
    """M2.4: one job renders 16:9, 9:16, and 1:1 from a single analysis pass."""

    def setUp(self):
        self._media_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._media_root, ignore_errors=True)
        self._settings = override_settings(MEDIA_ROOT=self._media_root)
        self._settings.enable()
        self.addCleanup(self._settings.disable)

        input_dir = Path(self._media_root) / "input"
        input_dir.mkdir(parents=True)
        shutil.copy(FIXTURE, input_dir / "demo_input.mp4")

    def test_one_job_renders_all_three_aspects(self):
        upload = VideoUpload.objects.create(
            file="input/demo_input.mp4", original_name="demo_input.mp4", size_bytes=FIXTURE.stat().st_size
        )
        job = PromoJob.objects.create(
            upload=upload,
            params={
                "aspect": "16:9",
                "aspects": ["16:9", "9:16", "1:1"],
                "config": {"target_duration": 15, "aspect": "16:9", "scene_threshold": 15.0},
            },
        )

        jobs.run_job(job.id)

        job.refresh_from_db()
        self.assertEqual(job.status, "done")
        outputs = list(job.outputs.all())
        self.assertEqual({o.aspect for o in outputs}, {"16:9", "9:16", "1:1"})

        # Scene/SegmentScore persisted once, not once per aspect.
        self.assertEqual(job.scenes.count(), job.scenes.values("index").distinct().count())

        for output in outputs:
            path = Path(output.file.path)
            self.assertTrue(path.exists())
            w, h = self._dimensions(path)
            self.assertAlmostEqual(w / h, self._expected_ratio(output.aspect), delta=0.05)

    @staticmethod
    def _dimensions(path: Path) -> tuple[int, int]:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        w, h = out.stdout.strip().split("x")
        return int(w), int(h)

    @staticmethod
    def _expected_ratio(aspect: str) -> float:
        return {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}[aspect]
