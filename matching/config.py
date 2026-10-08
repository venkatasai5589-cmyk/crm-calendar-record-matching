"""Settings for the matcher.

Weights and thresholds were chosen by reasoning about the data before the
evaluation was run (see README, "Key decisions"). They are not fitted to the labels.
"""

import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


def setup_logging() -> None:
    """Log to stderr. Set LOG_LEVEL=DEBUG to see every pair and data-quality issue."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

# Assumption: the firm's offices are NYC, DC and Boston, so naive timestamps are US/Eastern.
LOCAL_TZ = ZoneInfo("America/New_York")
FIRM_DOMAIN = "firma.com"

# Relative importance of each signal. Date, time and client identity separate
# "same meeting" from "same client, different meeting" (the labeled negatives).
WEIGHTS = {
    "same_date": 3.0,
    "time_proximity": 2.0,
    "client_attending": 3.0,
    "owner_attending": 1.0,
    "company_mentioned": 1.5,
    "title_similarity": 1.0,
    "location_similarity": 0.5,
}

MATCH_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.50

CANDIDATE_WINDOW_DAYS = 1
# Time proximity falls linearly from 1 (same start) to 0 (this many minutes apart).
# A labeled match (CRM-1016 / CAL-A17) is 120 minutes apart, so this must be generous.
TIME_DECAY_MINUTES = 240
DUPLICATE_MAX_GAP_MINUTES = 60

PLACEHOLDER_VALUES = {"multiple", "tbd", "n/a", "na", "unknown", "none", "-"}
VIRTUAL_KEYWORDS = ("zoom", "teams", "virtual", "webex", "google meet")

STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "with", "to", "in", "on", "at", "re", "call",
    "meeting", "discussion", "session", "https", "www", "com", "us", "j",
}
# Words too common in company names to identify a company on their own.
GENERIC_COMPANY_WORDS = {
    "capital", "group", "partners", "holdings", "advisors", "ventures", "investments",
    "investors", "wealth", "institutional", "fund", "management", "llc", "inc",
}
