"""
Unit Tests for MoodMentor Core Modules
"""

import unittest
import sys
from pathlib import Path
import tempfile
import sqlite3

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_classifier import EmotionClassifier
from src.recommender import WellBeingRecommender
from src.database import MoodDatabase


class TestMoodMentor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotion_classifier = EmotionClassifier()
        self.recommender = WellBeingRecommender()

        # Temporary database for isolated testing
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = MoodDatabase(db_path=Path(self.temp_db_file.name))

    def tearDown(self):
        try:
            Path(self.temp_db_file.name).unlink(missing_ok=True)
        except Exception:
            pass

    def test_preprocessor_empty(self):
        res = self.preprocessor.process("")
        self.assertTrue(res["is_empty"])
        self.assertEqual(res["stats"]["word_count"], 0)

    def test_preprocessor_cleaning(self):
        sample = "Check out http://test.com <b>stressed</b> and worried!!!"
        res = self.preprocessor.process(sample)
        self.assertFalse(res["is_empty"])
        self.assertNotIn("http", res["clean_text"])
        self.assertNotIn("<b>", res["clean_text"])
        self.assertIn("stressed", res["tokens"])

    def test_sentiment_vader(self):
        pos_text = "I feel so grateful, joyful, and happy today!"
        res_pos = self.sentiment_analyzer.analyze(pos_text)
        self.assertEqual(res_pos["overall_sentiment"], "Positive")
        self.assertGreater(res_pos["compound_score"], 0.05)

        neg_text = "I am feeling terribly stressed and upset."
        res_neg = self.sentiment_analyzer.analyze(neg_text)
        self.assertEqual(res_neg["overall_sentiment"], "Negative")
        self.assertLess(res_neg["compound_score"], -0.05)

    def test_emotion_classifier_example(self):
        sample = "I am feeling very stressed because of my exams. I cannot focus on anything."
        sent = self.sentiment_analyzer.analyze(sample)
        emo = self.emotion_classifier.classify(sample, sent["compound_score"])
        self.assertEqual(emo["primary_emotion"], "Stress / Worry")
        self.assertGreater(emo["confidence_percent"], 30.0)

    def test_recommendations(self):
        sample = "I am feeling very stressed because of my exams. I cannot focus on anything."
        sent = self.sentiment_analyzer.analyze(sample)
        emo = self.emotion_classifier.classify(sample, sent["compound_score"])
        recs = self.recommender.get_recommendations(
            emo["primary_emotion"],
            sent["overall_sentiment"],
            emo["intensity"],
            sample
        )
        self.assertGreaterEqual(len(recs["actions"]), 2)
        self.assertIn("disclaimer", recs)
        self.assertIn("guided_exercise", recs)

    def test_database_persistence(self):
        row_id = self.db.save_analysis(
            full_text="Feeling peaceful today.",
            overall_sentiment="Positive",
            compound_score=0.65,
            primary_emotion="Calm / Contentment",
            confidence_percent=82.0,
            intensity="High",
            word_count=4,
            recommendations=[{"title": "Rest", "action": "Take time off", "category": "Rest"}],
            source_type="unit_test"
        )
        self.assertGreater(row_id, 0)
        history = self.db.get_history(limit=5)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["primary_emotion"], "Calm / Contentment")


if __name__ == "__main__":
    unittest.main()
