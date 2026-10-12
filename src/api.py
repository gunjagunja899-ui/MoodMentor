"""
FastAPI Backend for MoodMentor
Exposes RESTful endpoints for sentiment analysis, emotion classification, and recommendations.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_classifier import EmotionClassifier
from src.recommender import WellBeingRecommender
from src.database import MoodDatabase

app = FastAPI(
    title="MoodMentor API",
    description="AI-Powered Emotional Tone Analyzer & Well-Being Recommendation Engine",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system components
preprocessor = TextPreprocessor()
sentiment_analyzer = SentimentAnalyzer()
emotion_classifier = EmotionClassifier()
recommender = WellBeingRecommender()
db = MoodDatabase()


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")
    source_type: Optional[str] = Field("api", description="Source identifier")
    save_to_db: Optional[bool] = Field(True, description="Whether to log entry into database")


class AnalyzeResponse(BaseModel):
    status: str
    overall_sentiment: str
    compound_score: float
    sentiment_breakdown: Dict[str, float]
    primary_emotion: str
    secondary_emotion: str
    emotion_confidence_percent: float
    emotional_intensity: str
    emotion_distribution: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    guided_exercise: Dict[str, Any]
    word_count: int
    clean_text: str
    disclaimer: str


@app.get("/")
def root():
    return {
        "app": "MoodMentor API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/history", "/analytics"]
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(payload: AnalyzeRequest):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace only.")

    # 1. Preprocessing
    prep_data = preprocessor.process(raw_text)

    # 2. Sentiment Analysis (VADER)
    sentiment_res = sentiment_analyzer.analyze(raw_text, prep_data["clean_text"])

    # 3. Emotion Classification
    emotion_res = emotion_classifier.classify(raw_text, sentiment_res["compound_score"])

    # 4. Recommendation Engine
    recs_res = recommender.get_recommendations(
        primary_emotion=emotion_res["primary_emotion"],
        overall_sentiment=sentiment_res["overall_sentiment"],
        intensity=emotion_res["intensity"],
        text=raw_text
    )

    # 5. Persist to Database if requested
    if payload.save_to_db:
        db.save_analysis(
            full_text=raw_text,
            overall_sentiment=sentiment_res["overall_sentiment"],
            compound_score=sentiment_res["compound_score"],
            primary_emotion=emotion_res["primary_emotion"],
            confidence_percent=emotion_res["confidence_percent"],
            intensity=emotion_res["intensity"],
            word_count=prep_data["stats"]["word_count"],
            recommendations=recs_res["actions"],
            source_type=payload.source_type or "api"
        )

    return AnalyzeResponse(
        status="success",
        overall_sentiment=sentiment_res["overall_sentiment"],
        compound_score=sentiment_res["compound_score"],
        sentiment_breakdown=sentiment_res["breakdown"],
        primary_emotion=emotion_res["primary_emotion"],
        secondary_emotion=emotion_res["secondary_emotion"],
        emotion_confidence_percent=emotion_res["confidence_percent"],
        emotional_intensity=emotion_res["intensity"],
        emotion_distribution=emotion_res["distribution"],
        recommendations=recs_res["actions"],
        guided_exercise=recs_res["guided_exercise"],
        word_count=prep_data["stats"]["word_count"],
        clean_text=prep_data["clean_text"],
        disclaimer=recs_res["disclaimer"]
    )


@app.get("/history")
def get_history(limit: int = Query(25, ge=1, le=100)):
    return {
        "status": "success",
        "count": limit,
        "records": db.get_history(limit=limit)
    }


@app.get("/analytics")
def get_analytics():
    return {
        "status": "success",
        "summary": db.get_summary_stats()
    }


@app.delete("/history")
def clear_history():
    db.clear_all()
    return {"status": "success", "message": "History cleared successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
