import shutil
import tempfile
from datetime import timedelta
from io import StringIO

from django.core.files.base import ContentFile
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from promoapp.models import PromoJob, PromoOutput, VideoUpload


class RetentionSweepTests(TestCase):
    def setUp(self):
        self._media_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._media_root, ignore_errors=True)
        settings_override = override_settings(MEDIA_ROOT=self._media_root)
        settings_override.enable()
        self.addCleanup(settings_override.disable)

    def test_old_input_deleted_recent_kept_db_rows_survive(self):
        old_upload = VideoUpload.objects.create(
            file=ContentFile(b"old", name="old.mp4"), original_name="old.mp4", size_bytes=3
        )
        old_upload.created_at = timezone.now() - timedelta(hours=25)
        old_upload.save(update_fields=["created_at"])

        recent_upload = VideoUpload.objects.create(
            file=ContentFile(b"recent", name="recent.mp4"), original_name="recent.mp4", size_bytes=6
        )

        call_command("retention_sweep", stdout=StringIO())

        old_upload.refresh_from_db()
        recent_upload.refresh_from_db()
        self.assertFalse(old_upload.file)
        self.assertTrue(recent_upload.file)
        # DB rows survive regardless of file deletion.
        self.assertTrue(VideoUpload.objects.filter(id=old_upload.id).exists())

    def test_old_output_deleted_but_scores_survive(self):
        upload = VideoUpload.objects.create(
            file=ContentFile(b"v", name="v.mp4"), original_name="v.mp4", size_bytes=1
        )
        job = PromoJob.objects.create(upload=upload, status="done")
        old_output = PromoOutput.objects.create(
            job=job, file=ContentFile(b"out", name="out.mp4"), aspect="16:9"
        )
        old_output.created_at = timezone.now() - timedelta(days=8)
        old_output.save(update_fields=["created_at"])

        call_command("retention_sweep", stdout=StringIO())

        old_output.refresh_from_db()
        self.assertFalse(old_output.file)
        self.assertTrue(PromoOutput.objects.filter(id=old_output.id).exists())
