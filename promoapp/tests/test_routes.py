from django.test import TestCase


class RoutesTests(TestCase):
    def test_home_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_generate_returns_200_on_get(self):
        response = self.client.get("/generate/")
        self.assertEqual(response.status_code, 200)
