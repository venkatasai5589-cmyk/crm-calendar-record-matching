"""Load both sources and normalize them into clean records.

Bad values are never dropped silently: each record keeps a list of the
data-quality issues found while normalizing it.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from matching.config import LOCAL_TZ, PLACEHOLDER_VALUES

logger = logging.getLogger(__name__)

# The first format is the expected one; the others are tolerated but flagged.
# Assumption: ambiguous dates are month-first, as the firm is US-based.
DATE_FORMATS = ["%Y-%m-%d", "%m-%d/%Y", "%m/%d/%Y", "%m-%d-%Y"]
FULL_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})?")
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]+")


@dataclass
class CrmRecord:
    id: str
    subject: str | None
    client_name: str | None
    client_company: str | None
    owner: str | None
    meeting_date: date | None
    start: datetime | None
    meeting_type: str | None
    location: str | None
    status: str | None
    issues: list[str] = field(default_factory=list)


@dataclass
class CalendarEvent:
    id: str
    title: str | None
    organizer: str | None
    attendees: list[str]
    start: datetime | None
    end: datetime | None
    location: str | None
    is_recurring: bool
    status: str | None
    issues: list[str] = field(default_factory=list)

    @property
    def meeting_date(self) -> date | None:
        return self.start.date() if self.start else None


def clean_text(value) -> str | None:
    """Strip whitespace and treat empty strings as missing."""
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def clean_name(value, issues: list[str]) -> str | None:
    name = clean_text(value)
    if name and name.lower() in PLACEHOLDER_VALUES:
        issues.append(f"placeholder_client_name: {name!r}")
        return None
    return name


def parse_date(value, issues: list[str]) -> date | None:
    text = clean_text(value)
    if text is None:
        issues.append("missing_date")
        return None
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        if fmt != DATE_FORMATS[0]:
            issues.append(f"nonstandard_date_format: {text!r}")
        return parsed
    issues.append(f"unparseable_date: {text!r}")
    return None


def parse_time(value, issues: list[str]):
    text = clean_text(value)
    if text is None:
        issues.append("missing_time")
        return None
    try:
        return datetime.strptime(text, "%H:%M").time()
    except ValueError:
        issues.append(f"unparseable_time: {text!r}")
        return None


def parse_timestamp(value, field_name: str, issues: list[str]) -> datetime | None:
    """Parse an ISO timestamp into naive local time."""
    text = clean_text(value)
    if text is None:
        issues.append(f"missing_{field_name}")
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        issues.append(f"unparseable_{field_name}: {text!r}")
        return None
    if not FULL_TIMESTAMP.fullmatch(text):
        issues.append(f"nonstandard_{field_name}_format: {text!r}")
    if parsed.tzinfo is not None:
        # All other timestamps are naive local times, so bring explicit UTC into the same frame.
        parsed = parsed.astimezone(LOCAL_TZ).replace(tzinfo=None)
        issues.append(f"{field_name}_converted_from_utc: {text!r}")
    return parsed


def clean_email(value, issues: list[str]) -> str | None:
    text = (clean_text(value) or "").lower()
    if "[at]" in text:
        text = text.replace("[at]", "@")
        issues.append(f"repaired_email: {value!r}")
    if not EMAIL.fullmatch(text):
        issues.append(f"invalid_attendee: {value!r}")
        return None
    return text


def parse_crm_record(raw: dict) -> CrmRecord:
    issues: list[str] = []
    meeting_date = parse_date(raw.get("meeting_date"), issues)
    meeting_time = parse_time(raw.get("meeting_time"), issues)
    start = datetime.combine(meeting_date, meeting_time) if meeting_date and meeting_time else None

    meeting_type = clean_text(raw.get("meeting_type"))
    client_name = clean_name(raw.get("client_name"), issues)
    if client_name is None and meeting_type != "Internal":
        issues.append("missing_client_name")

    return CrmRecord(
        id=raw["crm_id"],
        subject=clean_text(raw.get("subject")),
        client_name=client_name,
        client_company=clean_text(raw.get("client_company")),
        owner=clean_text(raw.get("relationship_owner")),
        meeting_date=meeting_date,
        start=start,
        meeting_type=meeting_type,
        location=clean_text(raw.get("location")),
        status=clean_text(raw.get("status")),
        issues=issues,
    )


def parse_calendar_event(raw: dict) -> CalendarEvent:
    issues: list[str] = []
    start = parse_timestamp(raw.get("start_time"), "start_time", issues)
    end = parse_timestamp(raw.get("end_time"), "end_time", issues)
    if start and end and end < start:
        issues.append("end_before_start")

    attendees = [email for value in raw.get("attendees") or [] if (email := clean_email(value, issues))]
    if not attendees:
        issues.append("no_attendees")

    return CalendarEvent(
        id=raw["event_id"],
        title=clean_text(raw.get("title")),
        organizer=clean_email(raw["organizer"], issues) if raw.get("organizer") else None,
        attendees=attendees,
        start=start,
        end=end,
        location=clean_text(raw.get("location")),
        is_recurring=bool(raw.get("is_recurring")),
        status=clean_text(raw.get("status")),
        issues=issues,
    )


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def log_loaded(path: Path, records: list[CrmRecord] | list[CalendarEvent]) -> None:
    flagged = [record for record in records if record.issues]
    logger.debug("Loaded %d records from %s (%d with data-quality issues)", len(records), path.name, len(flagged))
    for record in flagged:
        logger.debug("%s: %s", record.id, "; ".join(record.issues))


def load_crm(path: Path) -> list[CrmRecord]:
    records = [parse_crm_record(raw) for raw in load_json(path)]
    log_loaded(path, records)
    return records


def load_calendar(path: Path) -> list[CalendarEvent]:
    events = [parse_calendar_event(raw) for raw in load_json(path)]
    log_loaded(path, events)
    return events
