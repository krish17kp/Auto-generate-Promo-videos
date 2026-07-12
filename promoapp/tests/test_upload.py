from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings


class UploadSafetyTests(TestCase):
    def test_wrong_extension_rejected(self):
        upload = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
        response = self.client.post("/generate/", {"video": upload})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "unsupported file type")

    @override_settings(MAX_UPLOAD_MB=1)
    def test_oversize_file_rejected(self):
        upload = SimpleUploadedFile("big.mp4", b"0" * (2 * 1024 * 1024), content_type="video/mp4")
        response = self.client.post("/generate/", {"video": upload})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "file too large")

    def test_corrupt_video_rejected(self):
        upload = SimpleUploadedFile("broken.mp4", b"not a real video file", content_type="video/mp4")
        response = self.client.post("/generate/", {"video": upload})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "couldn&#x27;t read this video")

    def test_missing_file_rejected(self):
        response = self.client.post("/generate/", {})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "please choose a video file")
