"""
Database Module for MoodMentor
Manages analysis history and mood trends using SQLite / Relational Storage.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent.parent / "moodmentor_history.db"


class MoodDatabase:
    """
    SQLite database manager for storing emotional tone reports and retrieving analytics.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_FILE
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    text_snippet TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    overall_sentiment TEXT NOT NULL,
                    compound_score REAL NOT NULL,
                    primary_emotion TEXT NOT NULL,
                    confidence_percent REAL NOT NULL,
                    intensity TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    recommendations_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_analysis(
        self,
        full_text: str,
        overall_sentiment: str,
        compound_score: float,
        primary_emotion: str,
        confidence_percent: float,
        intensity: str,
        word_count: int,
        recommendations: List[Dict[str, Any]],
        source_type: str = "direct_input"
    ) -> int:
        """Saves a new analysis session to the database."""
        snippet = full_text[:120] + "..." if len(full_text) > 120 else full_text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_history (
                    timestamp, source_type, text_snippet, full_text,
                    overall_sentiment, compound_score, primary_emotion,
                    confidence_percent, intensity, word_count, recommendations_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                source_type,
                snippet,
                full_text,
                overall_sentiment,
                compound_score,
                primary_emotion,
                confidence_percent,
                intensity,
                word_count,
                json.dumps(recommendations)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent analysis records sorted by most recent first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analysis_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                r_dict = dict(row)
                try:
                    r_dict['recommendations'] = json.loads(r_dict.get('recommendations_json', '[]'))
                except Exception:
                    r_dict['recommendations'] = []
                results.append(r_dict)
            return results

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates statistical summary for the dashboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analysis_history")
            total_entries = cursor.fetchone()[0]

            if total_entries == 0:
                return {
                    "total_entries": 0,
                    "avg_compound": 0.0,
                    "sentiment_counts": {},
                    "emotion_counts": {}
                }

            cursor.execute("SELECT AVG(compound_score) FROM analysis_history")
            avg_compound = round(cursor.fetchone()[0] or 0.0, 3)

            cursor.execute("SELECT overall_sentiment, COUNT(*) FROM analysis_history GROUP BY overall_sentiment")
            sentiment_counts = dict(cursor.fetchall())

            cursor.execute("SELECT primary_emotion, COUNT(*) FROM analysis_history GROUP BY primary_emotion")
            emotion_counts = dict(cursor.fetchall())

            return {
                "total_entries": total_entries,
                "avg_compound": avg_compound,
                "sentiment_counts": sentiment_counts,
                "emotion_counts": emotion_counts
            }

    def clear_all(self):
        """Clears all records in history table."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_history")
            conn.commit()
