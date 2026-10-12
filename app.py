"""
MoodMentor - Streamlit Web Application
AI-Powered Emotional Tone Analyzer & Well-Being Recommendation System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
from pathlib import Path
import sys

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_classifier import EmotionClassifier
from src.recommender import WellBeingRecommender
from src.database import MoodDatabase

# Page Configuration
st.set_page_config(
    page_title="MoodMentor — AI Emotional Tone & Well-Being",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, glassmorphism, and responsive typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient & Hero styling */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #064e3b 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #67e8f9, #a7f3d0, #fef08a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 850px;
        line-height: 1.5;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .metric-hint {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    /* Recommendation Action Cards */
    .action-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        border-left: 5px solid #10b981;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .action-badge {
        display: inline-block;
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 0.4rem;
    }
    .action-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.25rem;
    }
    .action-desc {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.45;
    }

    /* Guided Exercise Card */
    .exercise-box {
        background: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.15) 0%, rgba(15, 23, 42, 0.6) 90%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 1.4rem;
        margin-top: 1rem;
    }
    .exercise-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #93c5fd;
        margin-bottom: 0.6rem;
    }

    /* Disclaimer box */
    .disclaimer-banner {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        font-size: 0.85rem;
        color: #fbbf24;
        margin-top: 1.8rem;
    }

    /* Note pills */
    .note-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-pos { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .pill-neg { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-neu { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
</style>
""", unsafe_allow_html=True)


# Initialize Session State & Components
@st.cache_resource
def get_components():
    return {
        "preprocessor": TextPreprocessor(),
        "sentiment_analyzer": SentimentAnalyzer(),
        "emotion_classifier": EmotionClassifier(),
        "recommender": WellBeingRecommender(),
        "db": MoodDatabase()
    }


components = get_components()
preprocessor = components["preprocessor"]
sentiment_analyzer = components["sentiment_analyzer"]
emotion_classifier = components["emotion_classifier"]
recommender = components["recommender"]
db = components["db"]

# Sidebar Navigation & Context
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.title("MoodMentor 🧠")
    st.caption("AI-Powered Emotional Tone Analyzer & Well-Being Recommendation System")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Information")
    st.info(
        "**Technology Stack:**\n"
        "- **Core:** Python, NLTK, spaCy\n"
        "- **Sentiment:** VADER\n"
        "- **Emotion:** Transformer/Lexical Hybrid\n"
        "- **Web UI:** Streamlit & Plotly\n"
        "- **Database:** SQLite / PostgreSQL Ready\n"
        "- **API:** FastAPI REST Backend"
    )

    st.markdown("---")
    st.markdown("### ⚠️ Clinical Boundary")
    st.caption(
        "MoodMentor estimates emotional tone for self-awareness and mindful productivity. "
        "It does **not** diagnose psychological illnesses."
    )

# Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">MoodMentor 🧠</div>
    <div class="hero-subtitle">
        Understand your emotional tone through natural language processing. Gain instant visibility into sentiment, detect stress patterns, and receive evidence-based well-being actions.
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_text, tab_file, tab_history, tab_api, tab_roadmap = st.tabs([
    "✍️ Text Analysis",
    "📁 Upload File (.txt / .csv)",
    "📊 History & Trend Analytics",
    "🔌 REST API Docs",
    "🗺️ Project Architecture & Roadmap"
])


# -------------------------------------------------------------
# TAB 1: SINGLE TEXT ANALYSIS
# -------------------------------------------------------------
with tab_text:
    st.subheader("Interactive Text Tone Analyzer")
    st.write("Type or paste any journal reflection, email draft, message, or exam thoughts:")

    # Quick Sample Fillers
    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
    preset_choice = None
    with col_preset1:
        if st.button("📝 Exam Stress Sample", use_container_width=True):
            preset_choice = "I am feeling very stressed because of my exams. I cannot focus on anything and deadlines are approaching fast."
    with col_preset2:
        if st.button("✨ Joy & Gratitude Sample", use_container_width=True):
            preset_choice = "Had an incredible morning walk by the river today! Feeling deeply grateful, energized, and fulfilled."
    with col_preset3:
        if st.button("⚡ Frustration Sample", use_container_width=True):
            preset_choice = "The team cancelled my presentation at the last minute. I feel so frustrated, annoyed, and ignored."
    with col_preset4:
        if st.button("🌿 Peaceful Evening Sample", use_container_width=True):
            preset_choice = "Finished all tasks for the day. Sitting with a warm cup of herbal tea feeling calm, peaceful, and grounded."

    # Input Area
    user_input = st.text_area(
        label="Input Text",
        value=preset_choice if preset_choice else "",
        placeholder="e.g. 'I am feeling very stressed because of my exams. I cannot focus on anything.'",
        height=140,
        help="Input at least 1 word for analysis."
    )

    col_btn, col_chk = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Emotional Tone", type="primary", use_container_width=True)
    with col_chk:
        save_db_checked = st.checkbox("Save this analysis session to history database", value=True)

    if analyze_clicked:
        if not user_input or not user_input.strip():
            st.error("⚠️ Empty input! Please enter or paste some text before clicking Analyze.")
        else:
            with st.spinner("Analyzing text with NLTK Preprocessing, VADER & Emotion Model..."):
                time.sleep(0.3)
                
                # 1. Preprocessing
                prep_result = preprocessor.process(user_input)
                
                # 2. Sentiment Analysis (VADER)
                sent_result = sentiment_analyzer.analyze(user_input, prep_result["clean_text"])
                
                # 3. Emotion Classification
                emo_result = emotion_classifier.classify(user_input, sent_result["compound_score"])
                
                # 4. Recommendations
                recs_result = recommender.get_recommendations(
                    primary_emotion=emo_result["primary_emotion"],
                    overall_sentiment=sent_result["overall_sentiment"],
                    intensity=emo_result["intensity"],
                    text=user_input
                )
                
                # 5. Save to database if requested
                if save_db_checked:
                    db.save_analysis(
                        full_text=user_input,
                        overall_sentiment=sent_result["overall_sentiment"],
                        compound_score=sent_result["compound_score"],
                        primary_emotion=emo_result["primary_emotion"],
                        confidence_percent=emo_result["confidence_percent"],
                        intensity=emo_result["intensity"],
                        word_count=prep_result["stats"]["word_count"],
                        recommendations=recs_result["actions"],
                        source_type="direct_input"
                    )

            st.markdown("---")
            st.markdown("### 📋 Text Analysis Report")

            # Metric Cards Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            # Sentiment card
            sentiment_label = sent_result["overall_sentiment"]
            pill_class = "pill-pos" if sentiment_label == "Positive" else ("pill-neg" if sentiment_label == "Negative" else "pill-neu")
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Overall Sentiment</div>
                    <div class="metric-value"><span class="note-pill {pill_class}">{sentiment_label}</span></div>
                    <div class="metric-hint">Polarity classification</div>
                </div>
                """, unsafe_allow_html=True)

            # VADER compound score card
            comp = sent_result["compound_score"]
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">VADER Compound Score</div>
                    <div class="metric-value">{comp:+.3f}</div>
                    <div class="metric-hint">Scale from -1.0 to +1.0</div>
                </div>
                """, unsafe_allow_html=True)

            # Detected emotion card
            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Detected Emotion</div>
                    <div class="metric-value" style="font-size:1.45rem;">{emo_result['primary_emotion']}</div>
                    <div class="metric-hint">Primary affective tone</div>
                </div>
                """, unsafe_allow_html=True)

            # Emotion confidence card (clarifying difference from compound score)
            with m_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Confidence & Intensity</div>
                    <div class="metric-value">{emo_result['confidence_percent']}%</div>
                    <div class="metric-hint">Intensity: <b>{emo_result['intensity']}</b> (Distinct from Compound)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visualizations & Insights Row
            vis_col1, vis_col2 = st.columns([1, 1])

            with vis_col1:
                st.markdown("#### 🎭 Emotion Distribution")
                dist = emo_result["distribution"]
                df_dist = pd.DataFrame({
                    "Emotion": list(dist.keys()),
                    "Probability": [v * 100 for v in dist.values()]
                }).sort_values(by="Probability", ascending=True)

                fig_bar = px.bar(
                    df_dist,
                    x="Probability",
                    y="Emotion",
                    orientation="h",
                    text=[f"{v:.1f}%" for v in df_dist["Probability"]],
                    color="Probability",
                    color_continuous_scale="Purples"
                )
                fig_bar.update_layout(
                    height=300,
                    margin=dict(l=0, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    xaxis=dict(title="Confidence (%)", range=[0, 105]),
                    yaxis=dict(title="")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with vis_col2:
                st.markdown("#### ⚖️ VADER Sentiment Breakdown")
                bd = sent_result["breakdown"]
                labels = ["Positive", "Neutral", "Negative"]
                values = [bd["positive"], bd["neutral"], bd["negative"]]
                colors = ["#22c55e", "#94a3b8", "#ef4444"]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.55,
                    marker=dict(colors=colors),
                    textinfo="label+percent"
                )])
                fig_donut.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # Suggested Well-Being Actions
            st.markdown("---")
            st.markdown("### 🌱 Suggested Well-Being Actions")
            st.caption("Customized, actionable steps recommended based on your detected emotional tone:")

            for action in recs_result["actions"]:
                st.markdown(f"""
                <div class="action-card">
                    <span class="action-badge">{action['category']}</span>
                    <div class="action-title">{action['title']}</div>
                    <div class="action-desc">{action['action']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Guided Exercise Section
            exercise = recs_result["guided_exercise"]
            st.markdown(f"""
            <div class="exercise-box">
                <div class="exercise-title">🧘 Guided Mindfulness: {exercise['name']}</div>
            """, unsafe_allow_html=True)
            for step in exercise["steps"]:
                st.markdown(f"- **{step}**")
            st.markdown("</div>", unsafe_allow_html=True)

            # Preprocessing Details Expander
            with st.expander("🔍 Inspect Text Preprocessing Details (NLTK Tokenization & Lemmatization)"):
                p_c1, p_c2, p_c3 = st.columns(3)
                p_c1.metric("Word Count", prep_result["stats"]["word_count"])
                p_c2.metric("Sentences", prep_result["stats"]["sentence_count"])
                p_c3.metric("Unique Keywords", prep_result["stats"]["unique_words"])

                st.write("**Cleaned Text (Symbols & URLs Removed):**")
                st.code(prep_result["clean_text"])

                st.write("**Lemmatized Tokens (Stop Words Filtered):**")
                st.write(prep_result["lemmatized_tokens"])

            # Medical Disclaimer
            st.markdown(f"""
            <div class="disclaimer-banner">
                {recs_result['disclaimer']}
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 2: FILE UPLOAD (.txt / .csv)
# -------------------------------------------------------------
with tab_file:
    st.subheader("Batch File Processing (.txt and .csv)")
    st.write("Upload a journal text file (`.txt`) or a batch dataset (`.csv`) with sentences to analyze.")

    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=["txt", "csv"],
        help="Upload .txt for a single document, or .csv containing a text column."
    )

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(f"📁 **Selected File:** `{uploaded_file.name}` ({uploaded_file.size} bytes)")

        if uploaded_file.name.endswith(".txt"):
            text_data = uploaded_file.read().decode("utf-8")
            st.text_area("File Content Preview", text_data, height=120)

            if st.button("🚀 Analyze Uploaded Document", type="primary"):
                with st.spinner("Processing document..."):
                    prep = preprocessor.process(text_data)
                    sent = sentiment_analyzer.analyze(text_data, prep["clean_text"])
                    emo = emotion_classifier.classify(text_data, sent["compound_score"])
                    recs = recommender.get_recommendations(
                        primary_emotion=emo["primary_emotion"],
                        overall_sentiment=sent["overall_sentiment"],
                        intensity=emo["intensity"],
                        text=text_data
                    )
                    # Persist
                    db.save_analysis(
                        full_text=text_data,
                        overall_sentiment=sent["overall_sentiment"],
                        compound_score=sent["compound_score"],
                        primary_emotion=emo["primary_emotion"],
                        confidence_percent=emo["confidence_percent"],
                        intensity=emo["intensity"],
                        word_count=prep["stats"]["word_count"],
                        recommendations=recs["actions"],
                        source_type="file_txt"
                    )

                st.success("✅ Analysis Complete! Saved to History.")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Overall Sentiment", sent["overall_sentiment"])
                col_r2.metric("VADER Compound", f"{sent['compound_score']:+.3f}")
                col_r3.metric("Detected Emotion", emo["primary_emotion"])

        elif uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
                st.write("📊 Data Preview:")
                st.dataframe(df.head(), use_container_width=True)

                text_column = st.selectbox(
                    "Select the column containing text entries to analyze:",
                    options=df.columns
                )

                if st.button("🚀 Run Batch Analysis on CSV", type="primary"):
                    progress_bar = st.progress(0)
                    results_list = []

                    for idx, row in df.iterrows():
                        text_val = str(row[text_column])
                        prep = preprocessor.process(text_val)
                        sent = sentiment_analyzer.analyze(text_val, prep["clean_text"])
                        emo = emotion_classifier.classify(text_val, sent["compound_score"])
                        recs = recommender.get_recommendations(
                            primary_emotion=emo["primary_emotion"],
                            overall_sentiment=sent["overall_sentiment"],
                            intensity=emo["intensity"],
                            text=text_val
                        )

                        results_list.append({
                            "Text": text_val[:80] + "..." if len(text_val) > 80 else text_val,
                            "Sentiment": sent["overall_sentiment"],
                            "Compound Score": sent["compound_score"],
                            "Emotion": emo["primary_emotion"],
                            "Confidence %": emo["confidence_percent"],
                            "Intensity": emo["intensity"],
                            "Top Recommendation": recs["actions"][0]["title"] if recs["actions"] else "Take a breath"
                        })

                        # Save to database
                        db.save_analysis(
                            full_text=text_val,
                            overall_sentiment=sent["overall_sentiment"],
                            compound_score=sent["compound_score"],
                            primary_emotion=emo["primary_emotion"],
                            confidence_percent=emo["confidence_percent"],
                            intensity=emo["intensity"],
                            word_count=prep["stats"]["word_count"],
                            recommendations=recs["actions"],
                            source_type="file_csv"
                        )
                        progress_bar.progress((idx + 1) / len(df))

                    res_df = pd.DataFrame(results_list)
                    st.success(f"🎉 Successfully analyzed {len(res_df)} rows and logged them to history!")
                    st.dataframe(res_df, use_container_width=True)

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Analyzed CSV Report",
                        data=csv_export,
                        file_name="moodmentor_analyzed_report.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")


# -------------------------------------------------------------
# TAB 3: HISTORY & TREND ANALYTICS
# -------------------------------------------------------------
with tab_history:
    st.subheader("Emotional Trends & History Dashboard")
    st.write("Track emotional patterns, sentiment trajectories, and well-being logs across time.")

    stats = db.get_summary_stats()
    records = db.get_history(limit=100)

    if stats["total_entries"] == 0:
        st.info("ℹ️ No analysis entries found yet. Run an analysis in Tab 1 or Tab 2 to start tracking trends!")
    else:
        # Overview Cards
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        s_c1.metric("Total Entries Logged", stats["total_entries"])
        s_c2.metric("Average Compound Score", f"{stats['avg_compound']:+.3f}")
        
        top_sentiment = max(stats["sentiment_counts"], key=stats["sentiment_counts"].get) if stats["sentiment_counts"] else "N/A"
        top_emotion = max(stats["emotion_counts"], key=stats["emotion_counts"].get) if stats["emotion_counts"] else "N/A"
        
        s_c3.metric("Dominant Sentiment", top_sentiment)
        s_c4.metric("Most Frequent Emotion", top_emotion)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        c_col1, c_col2 = st.columns([1, 1])

        df_history = pd.DataFrame(records)

        with c_col1:
            st.markdown("#### 📈 Sentiment Compound Trajectory")
            if "timestamp" in df_history.columns and "compound_score" in df_history.columns:
                df_history['id_rev'] = range(len(df_history), 0, -1)
                df_plot = df_history.sort_values(by="id")

                fig_line = px.line(
                    df_plot,
                    x="timestamp",
                    y="compound_score",
                    markers=True,
                    color="overall_sentiment",
                    color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"},
                    title="Mood Arc Over Time"
                )
                fig_line.add_hline(y=0.0, line_dash="dash", line_color="#64748b", annotation_text="Neutral Line")
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    margin=dict(l=0, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_line, use_container_width=True)

        with c_col2:
            st.markdown("#### 🎯 Emotion Frequency Distribution")
            emo_counts_df = pd.DataFrame(list(stats["emotion_counts"].items()), columns=["Emotion", "Count"])
            fig_pie = px.pie(
                emo_counts_df,
                names="Emotion",
                values="Count",
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📜 Recent Analysis Logs")
        
        display_cols = ["id", "timestamp", "overall_sentiment", "compound_score", "primary_emotion", "confidence_percent", "intensity", "text_snippet"]
        st.dataframe(
            df_history[display_cols].rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "overall_sentiment": "Sentiment",
                "compound_score": "Compound",
                "primary_emotion": "Emotion",
                "confidence_percent": "Confidence %",
                "intensity": "Intensity",
                "text_snippet": "Snippet"
            }),
            use_container_width=True
        )

        col_clr, col_space = st.columns([1, 4])
        with col_clr:
            if st.button("🗑️ Clear History Database"):
                db.clear_all()
                st.rerun()


# -------------------------------------------------------------
# TAB 4: REST API DOCUMENTATION & TESTING
# -------------------------------------------------------------
with tab_api:
    st.subheader("MoodMentor RESTful API (FastAPI)")
    st.write("MoodMentor includes a production-grade FastAPI backend for third-party integrations.")

    st.markdown("""
    #### 🚀 How to Start the FastAPI Server:
    Run the following command in your terminal:
    ```bash
    python src/api.py
    # OR
    uvicorn src.api:app --reload --port 8000
    ```
    Once running, interactive Swagger docs are accessible at: `http://127.0.0.1:8000/docs`

    #### 📌 Key Endpoints:
    | Method | Endpoint | Description |
    | :--- | :--- | :--- |
    | `GET` | `/` | Service health status check |
    | `POST` | `/analyze` | Analyze text sentiment, emotions & recommendations |
    | `GET` | `/history` | Fetch recent analysis records |
    | `GET` | `/analytics` | Retrieve emotional trend metrics |
    | `DELETE` | `/history` | Clear history logs |

    #### 💡 Example Request (Python):
    ```python
    import requests

    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        json={"text": "I am feeling stressed about exams and cannot focus."}
    )
    print(response.json())
    ```
    """)


# -------------------------------------------------------------
# TAB 5: PROJECT ARCHITECTURE & PPT ROADMAP
# -------------------------------------------------------------
with tab_roadmap:
    st.subheader("MoodMentor Project Architecture & PPT Roadmap")

    st.markdown("""
    ### 🏗️ Technical Architecture
    ```
    User Text / File Upload (.txt, .csv)
                    │
                    ▼
    ┌─────────────────────────────────┐
    │     1. Text Preprocessing       │ ──► [NLTK Tokenizer, Lemmatizer, Stopwords Filter]
    └─────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    ┌────────────────┐    ┌──────────────────────────┐
    │ 2. VADER       │    │ 3. Emotion Classifier    │
    │ Sentiment      │    │ (DistilBERT / Hybrid)    │
    │ Polarity       │    │ Granular Emotion & Score │
    └────────────────┘    └──────────────────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
    ┌─────────────────────────────────┐
    │ 4. Recommendation Engine        │ ──► [Tailored Actionable Wellness Suggestions]
    └─────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    ┌────────────────┐    ┌──────────────────────────┐
    │ 5. Database    │    │ 6. Streamlit & FastAPI   │
    │ (SQLite/Postgr)│    │ Interactive Web UI & API │
    └────────────────┘    └──────────────────────────┘
    ```

    ### 🗺️ PPT Milestones Progress:
    - ✅ **Milestone 1 (Basic Working Project)**: Text cleaning, tokenization, stop-words, lemmatization, VADER sentiment & compound score, input validations.
    - ✅ **Milestone 2 (Deep Emotion Analysis)**: Multi-emotion classification (Stress, Joy, Sadness, Anger, Fear, Calm, Neutral), confidence scoring & intensity.
    - ✅ **Milestone 3 (Recommendations & Backend)**: Contextual recommendation engine, SQLite/PostgreSQL schema, FastAPI REST API, trend analytics.
    - ✅ **Milestone 4 (Integration & Delivery)**: Streamlit interactive UI, CLI command runner, unit test suite, sample datasets, complete documentation.

    ### ⚠️ Crucial Concept: Sentiment vs Emotion Confidence
    - **VADER Compound Score:** A normalized valence score between **-1.0 (extreme negative)** and **+1.0 (extreme positive)** representing overall polarity.
    - **Emotion Confidence:** A probability percentage (**0% to 100%**) denoting how strongly a specific affective category (e.g. Stress / Worry) was recognized.
    """)
