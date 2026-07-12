import unittest

from promoapp.eval_ablation import run_ablation


class HybridBeatsBaselinesTests(unittest.TestCase):
    """Locks in the M2.2 gate: hybrid must beat every single-signal baseline on precision@5."""

    @classmethod
    def setUpClass(cls):
        cls.results = run_ablation()

    def test_hybrid_beats_audio_only(self):
        self.assertGreater(self.results["hybrid"]["precision_at_5"], self.results["audio_only"]["precision_at_5"])

    def test_hybrid_beats_clip_only(self):
        self.assertGreater(self.results["hybrid"]["precision_at_5"], self.results["clip_only"]["precision_at_5"])

    def test_hybrid_beats_motion_only(self):
        self.assertGreater(self.results["hybrid"]["precision_at_5"], self.results["motion_only"]["precision_at_5"])
