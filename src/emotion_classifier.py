"""
Emotion Classification Module for MoodMentor
Identifies granular emotions (Stress/Worry, Sadness, Anger, Fear, Joy, Calm, Neutral),
computes model confidence scores (0.0 to 1.0), and ranks emotional distribution.
"""

from typing import Dict, List, Any, Tuple
import re
import math


class EmotionClassifier:
    """
    Granular emotion classifier tailored for well-being assessment.
    Uses multi-dimensional semantic mapping and emotion cue extraction,
    with an architecture designed to seamlessly bridge into DistilBERT/BERT models.
    """

    EMOTIONS = [
        "Stress / Worry",
        "Sadness / Grief",
        "Anger / Frustration",
        "Fear / Anxiety",
        "Joy / Happiness",
        "Calm / Contentment",
        "Neutral / Reflective"
    ]

    def __init__(self):
        # Lexicon of emotional stems, synonyms, and expressions
        self.emotion_lexicon = {
            "Stress / Worry": [
                "stress", "stressed", "stressing", "stressful", "worry", "worried", "worrying",
                "overwhelm", "overwhelmed", "overwhelming", "pressure", "pressured", "burden",
                "burdened", "exam", "exams", "deadline", "deadlines", "cannot focus", "cant focus",
                "unable to concentrate", "too much work", "exhausted", "burnout", "burnt out",
                "hectic", "chaos", "chaotic", "restless", "panic", "panicking", "struggling",
                "fatigue", "tired", "sleepless", "insomnia", "rushing"
            ],
            "Sadness / Grief": [
                "sad", "sadness", "unhappy", "depressed", "depression", "miserable", "sorrow",
                "crying", "cried", "cry", "tears", "hopeless", "heartbroken", "lonely", "alone",
                "isolated", "empty", "gloomy", "down", "disappointed", "grief", "mourning",
                "lost", "hurt", "painful", "despair", "blue", "helpless", "worthless"
            ],
            "Anger / Frustration": [
                "angry", "anger", "furious", "mad", "annoyed", "annoying", "irritated",
                "frustrated", "frustration", "rage", "hate", "resent", "resentful", "outraged",
                "disgusted", "bitter", "agitated", "fuming", "pissed", "fed up", "infuriating"
            ],
            "Fear / Anxiety": [
                "fear", "afraid", "scared", "terrified", "anxious", "anxiety", "nervous",
                "frightened", "dread", "dreading", "uneasy", "jittery", "paranoid", "phobia",
                "trembling", "alarmed", "apprehensive", "insecure", "vulnerable"
            ],
            "Joy / Happiness": [
                "happy", "happiness", "joy", "joyful", "excited", "excitement", "glad",
                "cheerful", "delighted", "thrilled", "blessed", "wonderful", "amazing",
                "great", "fantastic", "elated", "ecstatic", "proud", "grateful", "gratitude",
                "loved", "loving", "optimistic", "fun", "celebrating", "achieved", "success"
            ],
            "Calm / Contentment": [
                "calm", "peace", "peaceful", "relaxed", "relaxing", "content", "contentment",
                "serene", "tranquil", "soothing", "grounded", "mindful", "rested", "balanced",
                "stable", "composed", "collected", "satisfied", "relief", "relieved", "hopeful"
            ]
        }

        # Weight modifiers for intensifiers
        self.intensifiers = {
            "very": 1.5,
            "extremely": 2.0,
            "so": 1.4,
            "really": 1.4,
            "completely": 1.6,
            "totally": 1.6,
            "deeply": 1.6,
            "terribly": 1.8,
            "immensely": 1.8,
            "too": 1.4,
            "super": 1.5
        }

    def _calculate_lexical_scores(self, text: str) -> Dict[str, float]:
        """Calculates raw emotion match scores using lexicon and contextual intensifiers."""
        text_lower = text.lower()
        words = re.findall(r'\b[a-z\'-]+\b', text_lower)
        scores = {emotion: 0.0 for emotion in self.EMOTIONS}

        # Contextual window search for intensifiers
        for i, word in enumerate(words):
            multiplier = 1.0
            if i > 0 and words[i - 1] in self.intensifiers:
                multiplier = self.intensifiers[words[i - 1]]

            for emotion, keywords in self.emotion_lexicon.items():
                for kw in keywords:
                    if " " in kw:
                        # multi-word phrase matching
                        if kw in text_lower:
                            scores[emotion] += 2.5 * multiplier
                    elif word == kw:
                        scores[emotion] += 1.5 * multiplier

        # Base neutral weight if almost no emotional words triggered
        total_raw = sum(scores.values())
        if total_raw < 0.5:
            scores["Neutral / Reflective"] = 1.5
        else:
            scores["Neutral / Reflective"] = 0.2

        return scores

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert raw scores into normalized probability distribution (0.0 to 1.0)."""
        temp = 1.2
        exp_scores = {k: math.exp(v / temp) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        return {k: round(v / total_exp, 4) for k, v in exp_scores.items()}

    def classify(self, text: str, compound_sentiment: float = 0.0) -> Dict[str, Any]:
        """
        Classifies emotional tone in text.
        Returns:
            primary_emotion: Strongest detected emotion
            secondary_emotion: Second highest emotion
            confidence: Probability score between 0.0 and 1.0
            intensity: Emotional intensity ("Low", "Medium", "High")
            distribution: Full dictionary of emotion probabilities
        """
        if not text or not text.strip():
            return {
                "primary_emotion": "Neutral / Reflective",
                "secondary_emotion": "None",
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "intensity": "Minimal",
                "distribution": {e: 0.0 for e in self.EMOTIONS}
            }

        raw_scores = self._calculate_lexical_scores(text)

        # Modulate scores with sentiment compound polarity
        if compound_sentiment < -0.3:
            raw_scores["Stress / Worry"] *= 1.3
            raw_scores["Sadness / Grief"] *= 1.2
            raw_scores["Fear / Anxiety"] *= 1.2
        elif compound_sentiment > 0.3:
            raw_scores["Joy / Happiness"] *= 1.3
            raw_scores["Calm / Contentment"] *= 1.2

        distribution = self._softmax(raw_scores)
        sorted_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        primary_emotion, primary_prob = sorted_emotions[0]
        secondary_emotion, secondary_prob = sorted_emotions[1] if len(sorted_emotions) > 1 else ("None", 0.0)

        # Determine intensity based on confidence and sentiment extremity
        if primary_prob >= 0.45 or abs(compound_sentiment) >= 0.65:
            intensity = "High"
        elif primary_prob >= 0.25 or abs(compound_sentiment) >= 0.35:
            intensity = "Medium"
        else:
            intensity = "Low"

        return {
            "primary_emotion": primary_emotion,
            "secondary_emotion": secondary_emotion,
            "confidence": primary_prob,
            "confidence_percent": round(primary_prob * 100, 1),
            "intensity": intensity,
            "distribution": distribution,
            "all_ranked": sorted_emotions
        }
