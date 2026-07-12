import unittest
from pathlib import Path
from unittest import mock

from promoapp.pipeline.features import transcript
from promoapp.pipeline.scenes import Scene

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "transcript_demo.mp4"


class TranscriptSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segments = transcript.transcribe(FIXTURE)

    def test_transcribes_spoken_content(self):
        self.assertEqual(len(self.segments), 3)
        joined = " ".join(s["text"].lower() for s in self.segments)
        self.assertIn("wondered", joined)
        self.assertIn("regular sentence", joined)
        self.assertIn("three", joined)

    def test_question_scores_highest(self):
        scores = {s["text"].strip(): transcript.hook_phrase_score(s["text"]) for s in self.segments}
        question = next(v for k, v in scores.items() if k.endswith("?"))
        mundane = next(v for k, v in scores.items() if "regular sentence" in k.lower())
        numbered = next(v for k, v in scores.items() if "three" in k.lower())
        self.assertGreater(question, numbered)
        self.assertGreater(numbered, mundane)
        self.assertEqual(mundane, 0.0)

    def test_best_hook_segment_picks_the_question(self):
        best = transcript.best_hook_segment(self.segments)
        self.assertTrue(best["text"].strip().endswith("?"))

    def test_scene_transcript_scores_aggregates_by_overlap(self):
        scenes = [Scene(index=0, start_s=s["start"], end_s=s["end"]) for s in self.segments]
        scores = transcript.scene_transcript_scores(self.segments, scenes)
        self.assertEqual(len(scores), 3)
        self.assertEqual(float(scores.max()), transcript.hook_phrase_score(self.segments[0]["text"]))


class LlmHookPickerFallbackTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"start": 0.0, "end": 2.0, "text": "This is a regular sentence."},
            {"start": 2.0, "end": 4.0, "text": "Have you ever wondered what happens next?"},
        ]

    def test_falls_back_to_keyword_heuristic_without_api_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            result = transcript.llm_hook_pick(self.segments, api_key=None)
        self.assertEqual(result, transcript.best_hook_segment(self.segments))

    def test_falls_back_on_api_error(self):
        with mock.patch("openai.OpenAI", side_effect=RuntimeError("network down")):
            result = transcript.llm_hook_pick(self.segments, api_key="fake-key")
        self.assertEqual(result, transcript.best_hook_segment(self.segments))

    def test_empty_segments_returns_none(self):
        self.assertIsNone(transcript.llm_hook_pick([], api_key="fake-key"))
