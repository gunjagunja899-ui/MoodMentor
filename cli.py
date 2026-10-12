"""
CLI Interface for MoodMentor
Allows quick emotional tone analysis and well-being recommendation directly from terminal.
"""

import argparse
import sys
from pathlib import Path

# Enable UTF-8 safe output for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_classifier import EmotionClassifier
from src.recommender import WellBeingRecommender
from src.database import MoodDatabase


def main():
    parser = argparse.ArgumentParser(
        description="MoodMentor CLI - AI Emotional Tone Analyzer & Well-Being Recommender"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", type=str, help="Direct text string to analyze")
    group.add_argument("--file", "-f", type=str, help="Path to text or csv file to analyze")

    parser.add_argument("--save", action="store_true", help="Save result to database history")
    args = parser.parse_args()

    content = ""
    source = "cli_direct"

    if args.text:
        content = args.text
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[Error] File not found: {args.file}")
            sys.exit(1)
        try:
            content = file_path.read_text(encoding="utf-8")
            source = f"cli_file_{file_path.suffix}"
        except Exception as e:
            print(f"[Error] Failed to read file: {e}")
            sys.exit(1)

    if not content.strip():
        print("[Error] Input text is empty!")
        sys.exit(1)

    preprocessor = TextPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()
    emotion_classifier = EmotionClassifier()
    recommender = WellBeingRecommender()
    db = MoodDatabase()

    print("\n" + "=" * 60)
    print("🧠 MOODMENTOR — TEXT ANALYSIS REPORT")
    print("=" * 60)

    # Preprocessing
    prep = preprocessor.process(content)
    print(f"\n[Text Stats] Words: {prep['stats']['word_count']} | Sentences: {prep['stats']['sentence_count']}")

    # Sentiment
    sent = sentiment_analyzer.analyze(content, prep["clean_text"])
    print(f"\n[Overall Sentiment]  : {sent['overall_sentiment']}")
    print(f"[VADER Compound Score]: {sent['compound_score']:+.4f} (Pos: {sent['breakdown']['positive']}, Neg: {sent['breakdown']['negative']}, Neu: {sent['breakdown']['neutral']})")
    print(f"[Sentiment Intensity]: {sent['intensity']}")

    # Emotion
    emo = emotion_classifier.classify(content, sent["compound_score"])
    print(f"\n[Detected Emotion]   : {emo['primary_emotion']}")
    print(f"[Emotion Confidence] : {emo['confidence_percent']}%")
    print(f"[Secondary Emotion]  : {emo['secondary_emotion']}")
    print(f"[Emotional Intensity]: {emo['intensity']}")

    # Recommendations
    recs = recommender.get_recommendations(
        primary_emotion=emo["primary_emotion"],
        overall_sentiment=sent["overall_sentiment"],
        intensity=emo["intensity"],
        text=content
    )

    print("\n🌱 SUGGESTED WELL-BEING ACTIONS:")
    for idx, act in enumerate(recs["actions"], 1):
        print(f"  {idx}. [{act['category']}] {act['title']}")
        print(f"     -> {act['action']}")

    print(f"\n🧘 GUIDED EXERCISE: {recs['guided_exercise']['name']}")
    for step in recs["guided_exercise"]["steps"]:
        print(f"   • {step}")

    print("\n" + "-" * 60)
    print(f"ℹ️  {recs['disclaimer']}")
    print("=" * 60 + "\n")

    if args.save:
        rec_id = db.save_analysis(
            full_text=content,
            overall_sentiment=sent["overall_sentiment"],
            compound_score=sent["compound_score"],
            primary_emotion=emo["primary_emotion"],
            confidence_percent=emo["confidence_percent"],
            intensity=emo["intensity"],
            word_count=prep["stats"]["word_count"],
            recommendations=recs["actions"],
            source_type=source
        )
        print(f"💾 Record successfully saved to history (ID: {rec_id})")


if __name__ == "__main__":
    main()
