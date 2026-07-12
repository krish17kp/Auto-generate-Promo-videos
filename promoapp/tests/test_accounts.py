from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from promoapp.models import PromoJob, VideoUpload


class SignupLoginTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        response = self.client.post(
            "/accounts/signup/",
            {"username": "alice", "password1": "correcthorsebatterystaple9", "password2": "correcthorsebatterystaple9"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username="alice").exists())
        self.assertIn("_auth_user_id", self.client.session)

    def test_login_page_loads(self):
        response = self.client.get("/accounts/login/")
        self.assertEqual(response.status_code, 200)


class HistoryTests(TestCase):
    def test_history_requires_login(self):
        response = self.client.get("/history/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_history_shows_only_own_jobs(self):
        alice = User.objects.create_user("alice", password="x")
        bob = User.objects.create_user("bob", password="x")

        alice_upload = VideoUpload.objects.create(owner=alice, original_name="alice.mp4", size_bytes=1)
        PromoJob.objects.create(upload=alice_upload, status="done")
        bob_upload = VideoUpload.objects.create(owner=bob, original_name="bob.mp4", size_bytes=1)
        PromoJob.objects.create(upload=bob_upload, status="done")

        self.client.force_login(alice)
        response = self.client.get("/history/")

        self.assertContains(response, "alice.mp4")
        self.assertNotContains(response, "bob.mp4")


@override_settings(RATELIMIT_ENABLE=True)
class RateLimitTests(TestCase):
    def test_generate_endpoint_blocks_after_limit(self):
        for _ in range(10):
            self.client.post("/generate/", {})

        response = self.client.post("/generate/", {})
        self.assertEqual(response.status_code, 403)
