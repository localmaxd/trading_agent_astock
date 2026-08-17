from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    FUNDAMENTALS = "fundamentals"
    TECHNICAL = "technical"
    GAME_THEORY = "game_theory"
    NEWS_SENTIMENT = "news_sentiment"
