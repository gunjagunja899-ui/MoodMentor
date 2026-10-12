"""
Sentiment Analysis Module for MoodMentor
Powered by NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner).
Extracts positive, negative, neutral proportions, compound score, and sentiment label.
"""

from typing import Dict, Any, List
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _ensure_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)


_ensure_vader()


class SentimentAnalyzer:
    """
    Analyzes emotional polarity and valence using VADER.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def _determine_intensity(self, compound: float) -> str:
        """Categorize absolute compound valence into intensity tiers."""
        abs_val = abs(compound)
        if abs_val >= 0.75:
            return "High / Strong"
        elif abs_val >= 0.40:
            return "Moderate"
        elif abs_val >= 0.05:
            return "Mild / Low"
        else:
            return "Neutral / Minimal"

    def analyze(self, text: str, clean_text: str = "") -> Dict[str, Any]:
        """
        Analyzes sentiment of provided text.
        VADER works well on text with punctuation, casing and emoticons,
        so we pass clean_text or raw_text appropriately.
        """
        target_text = clean_text if clean_text else text
        if not target_text or not target_text.strip():
            return {
                "overall_sentiment": "Neutral",
                "compound_score": 0.0,
                "intensity": "Neutral / Minimal",
                "breakdown": {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0
                },
                "explanation": "No text provided to evaluate."
            }

        scores = self.analyzer.polarity_scores(target_text)
        compound = scores['compound']
        pos = scores['pos']
        neg = scores['neg']
        neu = scores['neu']

        # Standard VADER classification thresholds
        if compound >= 0.05:
            overall = "Positive"
            explanation = (
                f"Text reflects predominantly positive tone (Compound: {compound:+.2f}). "
                f"Contains affirmative, encouraging, or constructive valence."
            )
        elif compound <= -0.05:
            overall = "Negative"
            explanation = (
                f"Text expresses noticeable distress, strain, or negative tone (Compound: {compound:+.2f})."
            )
        else:
            overall = "Neutral"
            explanation = (
                f"Text maintains an objective or balanced emotional tone (Compound: {compound:+.2f})."
            )

        intensity = self._determine_intensity(compound)

        return {
            "overall_sentiment": overall,
            "compound_score": round(compound, 4),
            "intensity": intensity,
            "breakdown": {
                "positive": round(pos, 4),
                "negative": round(neg, 4),
                "neutral": round(neu, 4)
            },
            "explanation": explanation
        }

    def analyze_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Analyzes sentiment for each sentence individually to visualize emotional arc.
        """
        results = []
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            scores = self.analyzer.polarity_scores(sentence)
            compound = scores['compound']
            if compound >= 0.05:
                sentiment = "Positive"
            elif compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            results.append({
                "sentence_index": idx + 1,
                "text": sentence,
                "sentiment": sentiment,
                "compound_score": round(compound, 4),
                "pos": round(scores['pos'], 3),
                "neg": round(scores['neg'], 3),
                "neu": round(scores['neu'], 3)
            })
        return results
