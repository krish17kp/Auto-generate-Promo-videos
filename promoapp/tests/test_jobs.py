import shutil
import tempfile
import threading
from pathlib import Path

from django.test import TransactionTestCase, override_settings

from promoapp import jobs
from promoapp.models import PromoJob, VideoUpload

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_input.mp4"


class JobLifecycleTests(TransactionTestCase):
    # TransactionTestCase, not TestCase: test_start_job_runs_in_background_thread spawns a
    # real thread that needs its own DB connection — TestCase's wrapping transaction would
    # deadlock a background thread's writes against the main thread's open transaction.
    def setUp(self):
        self._media_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._media_root, ignore_errors=True)
        self._settings = override_settings(MEDIA_ROOT=self._media_root)
        self._settings.enable()
        self.addCleanup(self._settings.disable)

        input_dir = Path(self._media_root) / "input"
        input_dir.mkdir(parents=True)
        shutil.copy(FIXTURE, input_dir / "sample_input.mp4")

    def _make_job(self, filename="sample_input.mp4", target_duration=15):
        upload = VideoUpload.objects.create(
            file=f"input/{filename}",
            original_name=filename,
            size_bytes=FIXTURE.stat().st_size,
        )
        return PromoJob.objects.create(
            upload=upload,
            params={"aspect": "16:9", "config": {"target_duration": target_duration, "aspect": "16:9"}},
        )

    def test_job_runs_queued_to_done(self):
        job = self._make_job()
        self.assertEqual(job.status, "queued")

        jobs.run_job(job.id)

        job.refresh_from_db()
        self.assertEqual(job.status, "done")
        self.assertEqual(job.progress, 100)
        self.assertIsNotNone(job.started_at)
        self.assertIsNotNone(job.finished_at)
        self.assertTrue(job.scenes.exists())
        self.assertTrue(job.outputs.exists())

    def test_job_failure_lands_as_failed_with_message(self):
        job = self._make_job(filename="does-not-exist.mp4")

        jobs.run_job(job.id)

        job.refresh_from_db()
        self.assertEqual(job.status, "failed")
        self.assertTrue(job.error_message)
        self.assertIsNotNone(job.finished_at)

    def test_start_job_runs_in_background_thread(self):
        job = self._make_job()

        before = set(threading.enumerate())
        jobs.start_job(job.id)
        spawned = set(threading.enumerate()) - before
        for thread in spawned:
            thread.join(timeout=30)

        job.refresh_from_db()
        self.assertEqual(job.status, "done")
