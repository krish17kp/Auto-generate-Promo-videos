import shutil
import tempfile
from pathlib import Path

from django.test import TestCase, override_settings

from promoapp import jobs
from promoapp.models import PromoJob, Scene, VideoUpload

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "demo_input.mp4"


class CapstoneScoringTests(TestCase):
    """M2.1 hybrid scoring + M2.3 narrative assembly, both live on one processed job."""

    def setUp(self):
        self._media_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._media_root, ignore_errors=True)
        self._settings = override_settings(MEDIA_ROOT=self._media_root)
        self._settings.enable()
        self.addCleanup(self._settings.disable)

        input_dir = Path(self._media_root) / "input"
        input_dir.mkdir(parents=True)
        shutil.copy(FIXTURE, input_dir / "demo_input.mp4")

        upload = VideoUpload.objects.create(
            file="input/demo_input.mp4", original_name="demo_input.mp4", size_bytes=FIXTURE.stat().st_size
        )
        self.job = PromoJob.objects.create(
            upload=upload,
            params={
                "aspect": "9:16",
                "config": {
                    "target_duration": 15, "aspect": "9:16", "scene_threshold": 15.0,
                    "use_clip": True, "use_motion": True, "use_quality": True, "full_narrative": True,
                    "title": "My Promo", "cta": "Watch Now",
                },
            },
        )
        jobs.run_job(self.job.id)
        self.job.refresh_from_db()

    def test_job_completes(self):
        self.assertEqual(self.job.status, "done")

    def test_segment_scores_carry_all_signals(self):
        scenes = Scene.objects.filter(job=self.job).select_related("score")
        self.assertGreaterEqual(scenes.count(), 3)
        for scene in scenes:
            score = scene.score
            self.assertIsNotNone(score.visual)
            self.assertIsNotNone(score.audio)
            self.assertIsNotNone(score.motion)
            self.assertIsNotNone(score.quality)
            self.assertIsNotNone(score.fused)

    def test_narrative_roles_assigned_without_duplicate_scenes(self):
        selected = Scene.objects.filter(job=self.job, score__selected=True).select_related("score")
        self.assertTrue(selected.exists())

        roles = [s.score.narrative_role for s in selected]
        self.assertEqual(len(roles), len(set(roles)))  # no role reused
        self.assertTrue(set(roles).issubset({"hook", "build", "climax", "outro"}))

        scene_indexes = [s.index for s in selected]
        self.assertEqual(len(scene_indexes), len(set(scene_indexes)))  # no scene used twice

    def test_build_clips_are_chronological(self):
        build_scenes = (
            Scene.objects.filter(job=self.job, score__narrative_role="build")
            .order_by("index")
            .values_list("start_s", flat=True)
        )
        self.assertEqual(list(build_scenes), sorted(build_scenes))
