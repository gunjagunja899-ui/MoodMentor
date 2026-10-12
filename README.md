# MoodMentor 🧠https://moodmentor-uhppjexh6uysrpdhscfpv6.streamlit.app/
### AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.64-red.svg)](https://streamlit.io/)
[![Backend](https://img.shields.io/badge/FastAPI-0.142-green.svg)](https://fastapi.tiangolo.com/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20VADER-yellow.svg)](https://www.nltk.org/)

MoodMentor is an intelligent AI-assisted mental wellness support tool. It analyzes user-written text (journals, reflections, messages, exam thoughts) using Natural Language Processing (NLP), evaluates emotional polarity and fine-grained emotional tones, and offers evidence-based well-being recommendations and guided mindfulness exercises.

> **Important Boundary & Clinical Disclaimer**: MoodMentor estimates emotional tone for self-reflection and personal wellness. It **does not** diagnose psychological disorders or replace professional psychiatric or medical care.

---

## 🌟 Key Features

1. **Dual Input Modes**:
   - Direct text input with quick sample buttons (e.g. *Exam Stress*, *Joy & Gratitude*, *Frustration*, *Calm*).
   - Document upload supporting `.txt` journal files and batch `.csv` datasets with exportable reports.
2. **Text Preprocessing Pipeline**:
   - Text cleaning (HTML, URLs, excessive punctuation removal).
   - Sentence and word tokenization using NLTK.
   - Stop-word filtering and WordNet lemmatization.
3. **Sentiment & Emotion Analysis**:
   - **VADER Sentiment Analysis**: Positive, Neutral, Negative breakdown and Compound Score (-1.0 to +1.0).
   - **Granular Emotion Classifier**: Detects *Stress / Worry*, *Joy / Happiness*, *Sadness / Grief*, *Anger / Frustration*, *Fear / Anxiety*, and *Calm / Contentment*.
   - **Model Confidence Score (%)**: Explicitly separated from the VADER compound score.
4. **Contextual Recommendation Engine**:
   - Actionable lifestyle tips (e.g., Short brain breaks, Pomodoro chunking, 5-4-3-2-1 grounding, gratitude reflection).
   - Interactive guided breathwork exercises (Box Breathing, Centering Breath).
5. **Dashboard & History Tracking**:
   - SQLite relational database logging each analysis session.
   - Interactive Plotly visualizations for emotional distribution and mood trajectories over time.
6. **Developer Integrations**:
   - **FastAPI REST Backend**: Ready endpoints (`/analyze`, `/history`, `/analytics`) with auto-generated Swagger UI.
   - **CLI Mode**: Terminal command runner for quick automated checks.

---

## 📁 Project Architecture

```
AI-Powered Emotional Tone Analyzer and Well-Being Recommendation/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py        # Cleaning, tokenization, stop-words, lemmatization
│   ├── sentiment_analyzer.py  # VADER sentiment analysis & compound scores
│   ├── emotion_classifier.py  # Multi-emotion classification & confidence scoring
│   ├── recommender.py         # Tailored well-being suggestions & guided breathing
│   ├── database.py            # SQLite storage for history and trends
│   └── api.py                 # FastAPI REST API endpoints
├── sample_data/
│   ├── sample_journal.txt     # Sample journal reflection
│   └── sample_entries.csv     # Sample multi-row dataset for batch testing
├── tests/
│   ├── __init__.py
│   └── test_modules.py        # Automated test suite
├── app.py                     # Streamlit Interactive Web Application
├── cli.py                     # Terminal command-line tool
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

### 3. Launch the FastAPI Backend (Optional)
```bash
python src/api.py
# OR
uvicorn src.api:app --reload --port 8000
```
Interactive API documentation will be available at: **`http://127.0.0.1:8000/docs`**

### 4. Run via Terminal (CLI Mode)
Analyze direct text:
```bash
python cli.py --text "I am feeling very stressed because of my exams. I cannot focus on anything." --save
```

Analyze a text file:
```bash
python cli.py --file sample_data/sample_journal.txt --save
```

### 5. Run the Automated Test Suite
```bash
python tests/test_modules.py
```

---

## 📊 Core Concepts: Sentiment vs. Emotion Confidence

| Concept | Range | Meaning |
| :--- | :--- | :--- |
| **VADER Compound Score** | `-1.0` to `+1.0` | Overall polarity valence of the passage (`>= +0.05` is positive, `<= -0.05` is negative). |
| **Emotion Confidence** | `0%` to `100%` | Model certainty that a specific discrete emotion (e.g., *Stress / Worry*) is expressed. |

---

## 🗺️ PPT Milestones Mapping

- **Milestone 1 (Basic Working Project)**: Text cleaning, tokenization, stop-words removal, lemmatization, VADER sentiment, compound score, and empty input validation.
- **Milestone 2 (Deep Emotion Analysis)**: Multi-emotion classification (Stress, Joy, Sadness, Anger, Fear, Calm, Neutral), confidence scoring, and intensity indicators.
- **Milestone 3 (Recommendations & Backend)**: Recommendation engine, SQLite persistence, FastAPI endpoints, and analytics charts.
- **Milestone 4 (Final Integration)**: Streamlit interactive UI, CLI command runner, test suite, and packaging.
