"""
Well-Being Recommendation Engine for MoodMentor
Provides actionable, science-backed lifestyle, mindfulness, and coping suggestions
tailored to detected emotional tone, sentiment polarity, and contextual triggers.
"""

from typing import Dict, List, Any


class WellBeingRecommender:
    """
    Generates actionable wellness recommendations, mindfulness exercises,
    and self-care prompts based on emotion and sentiment analysis.
    """

    DISCLAIMER = (
        "Disclaimer: MoodMentor is an AI-assisted self-reflection and well-being support tool. "
        "It provides estimates based on text analysis and DOES NOT provide medical diagnosis or psychiatric treatment. "
        "If you are experiencing severe emotional distress or a crisis, please seek guidance from a qualified mental health professional."
    )

    RECOMMENDATIONS_MAP = {
        "Stress / Worry": [
            {
                "title": "Short Brain Break",
                "action": "Step away from your workstation or study desk for 5–10 minutes. Rest your eyes and disconnect from screens.",
                "category": "Rest & Recharge"
            },
            {
                "title": "5-Minute Breathing Exercise",
                "action": "Practice Box Breathing or 4-7-8 breathing: Inhale for 4 seconds, hold for 7 seconds, exhale slowly for 8 seconds.",
                "category": "Mindfulness"
            },
            {
                "title": "Task Micro-Chunking",
                "action": "Divide large tasks or exam preparation into bite-sized 20-minute chunks (Pomodoro). Focus solely on the first tiny step.",
                "category": "Productivity & Focus"
            },
            {
                "title": "Cognitive Brain-Dump",
                "action": "Write down your top 3 worries on a piece of paper to clear mental bandwidth, then prioritize just one.",
                "category": "Reflection"
            }
        ],
        "Sadness / Grief": [
            {
                "title": "Self-Compassion Check-in",
                "action": "Acknowledge your emotions gently without self-criticism. Give yourself permission to feel and heal at your own pace.",
                "category": "Self-Compassion"
            },
            {
                "title": "Connect with a Trusted Person",
                "action": "Share your feelings with a close friend, family member, or mentor. Human connection eases emotional weight.",
                "category": "Social Support"
            },
            {
                "title": "Gentle Movement & Sunlight",
                "action": "Take a 10-minute slow walk outdoors. Natural light and light physical movement help stimulate endorphins.",
                "category": "Physical Well-being"
            },
            {
                "title": "Expressive Journaling",
                "action": "Write freely about how you feel without editing yourself. Externalizing thoughts helps process sorrow.",
                "category": "Journaling"
            }
        ],
        "Anger / Frustration": [
            {
                "title": "Intentional Pause",
                "action": "Count to 10 or take 5 deep diaphragmatic breaths before responding to any trigger or email.",
                "category": "Emotional Regulation"
            },
            {
                "title": "Physical Tension Release",
                "action": "Drink a tall glass of cold water, do shoulder rolls, or go for a brisk walk to discharge physical adrenaline.",
                "category": "Somatic Relief"
            },
            {
                "title": "Sphere of Control Reframing",
                "action": "Ask yourself: 'What part of this situation is within my direct control?' Focus your energy solely on that slice.",
                "category": "Cognitive Reframing"
            }
        ],
        "Fear / Anxiety": [
            {
                "title": "5-4-3-2-1 Sensory Grounding",
                "action": "Identify 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                "category": "Grounding"
            },
            {
                "title": "Grounding Breathwork",
                "action": "Place one hand on your chest and one on your abdomen. Breathe deeply so only the lower hand moves.",
                "category": "Mindfulness"
            },
            {
                "title": "Fact vs Assumption Filter",
                "action": "Ask yourself: 'Is this fear an established certainty right now, or an anticipation created by anxiety?'",
                "category": "Cognitive Reframing"
            }
        ],
        "Joy / Happiness": [
            {
                "title": "Gratitude Anchor",
                "action": "Write down 3 specific details that contributed to this joyful experience so you can revisit it later.",
                "category": "Gratitude"
            },
            {
                "title": "Share the Radiance",
                "action": "Reach out to someone who supported you or share a kind word with a colleague or friend.",
                "category": "Connection"
            },
            {
                "title": "Mindful Savoring",
                "action": "Take 60 seconds to sit quietly and absorb how good this accomplishment or moment feels.",
                "category": "Mindfulness"
            }
        ],
        "Calm / Contentment": [
            {
                "title": "Protect Your Quiet Space",
                "action": "Preserve this peaceful state by maintaining healthy boundaries around screens and interruptions.",
                "category": "Boundaries"
            },
            {
                "title": "Reflective Reading or Hobby",
                "action": "Spend 20 minutes engaging in an uplifting book, creative hobby, or peaceful walk.",
                "category": "Enrichment"
            }
        ],
        "Neutral / Reflective": [
            {
                "title": "Body & Hydration Reset",
                "action": "Drink a glass of water, check your posture, and stretch your neck and shoulders.",
                "category": "Physical Reset"
            },
            {
                "title": "Set a Micro-Intention",
                "action": "Decide on one meaningful task or self-care action you wish to complete before your day concludes.",
                "category": "Clarity"
            }
        ]
    }

    def get_recommendations(
        self,
        primary_emotion: str,
        overall_sentiment: str,
        intensity: str,
        text: str = ""
    ) -> Dict[str, Any]:
        """
        Retrieves tailored well-being suggestions based on emotional state.
        """
        # Find matching emotion list or fallback to Neutral
        actions = self.RECOMMENDATIONS_MAP.get(primary_emotion, self.RECOMMENDATIONS_MAP["Neutral / Reflective"]).copy()

        # Contextual adaptations
        lower_text = text.lower() if text else ""
        contextual_tips = []

        if any(w in lower_text for w in ["exam", "study", "test", "assignment", "deadline"]):
            contextual_tips.append({
                "title": "Academic/Study Pacing",
                "action": "Adopt the Pomodoro technique (25 min study + 5 min break). Prioritize high-yield topics first and stop studying 1 hour before sleep.",
                "category": "Study Strategy"
            })

        if any(w in lower_text for w in ["sleep", "insomnia", "tired", "exhausted", "night"]):
            contextual_tips.append({
                "title": "Sleep Hygiene Protocol",
                "action": "Avoid screen exposure 30 minutes before bed. Keep your room dark and try chamomile tea or soothing audio.",
                "category": "Rest & Sleep"
            })

        # Prepend contextual tips if detected
        final_actions = contextual_tips + actions

        # Quick guided exercise
        if primary_emotion in ["Stress / Worry", "Fear / Anxiety", "Anger / Frustration"]:
            exercise = {
                "name": "Box Breathing (Samavritti)",
                "steps": [
                    "Step 1: Inhale smoothly through your nose for 4 counts.",
                    "Step 2: Hold your breath gently for 4 counts.",
                    "Step 3: Exhale slowly through your mouth for 4 counts.",
                    "Step 4: Hold empty lungs comfortably for 4 counts.",
                    "Repeat cycle 4 times."
                ]
            }
        elif primary_emotion in ["Sadness / Grief"]:
            exercise = {
                "name": "Self-Compassion Pause",
                "steps": [
                    "Step 1: Place a hand over your heart or on your arm.",
                    "Step 2: Feel the warmth and physical presence of your touch.",
                    "Step 3: Breathe in deeply, thinking: 'May I be kind to myself in this moment.'",
                    "Step 4: Breathe out slowly, letting go of self-blame."
                ]
            }
        else:
            exercise = {
                "name": "Gratitude & Centering Breath",
                "steps": [
                    "Step 1: Close your eyes and inhale deeply for 5 seconds.",
                    "Step 2: Note one person or moment you feel grateful for.",
                    "Step 3: Exhale gently with a gentle smile.",
                    "Step 4: Carry this centered calmness into your next task."
                ]
            }

        return {
            "primary_emotion": primary_emotion,
            "overall_sentiment": overall_sentiment,
            "intensity": intensity,
            "actions": final_actions[:4],
            "guided_exercise": exercise,
            "disclaimer": self.DISCLAIMER
        }
