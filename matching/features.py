"""Compare one CRM record with one calendar event.

Each signal is a number in [0, 1], or None when either side lacks the data.
None means "no evidence", so a missing field neither helps nor hurts a pair.
"""

import re

from matching.config import (
    FIRM_DOMAIN,
    GENERIC_COMPANY_WORDS,
    STOPWORDS,
    TIME_DECAY_MINUTES,
    VIRTUAL_KEYWORDS,
)
from matching.ingest import CalendarEvent, CrmRecord


def tokens(text: str | None) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {word for word in words if word not in STOPWORDS}


def overlap(a: set[str], b: set[str]) -> float | None:
    """Share of the smaller token set found in the larger one."""
    if not a or not b:
        return None
    return len(a & b) / min(len(a), len(b))


def email_local_part(name: str) -> str:
    """Guess a person's email local part: "Kevin O'Brien" -> "kevin.obrien"."""
    words = re.sub(r"[^a-z ]", "", name.lower()).split()
    return ".".join(words)


def calendar_people(event: CalendarEvent) -> set[str]:
    emails = set(event.attendees)
    if event.organizer:
        emails.add(event.organizer)
    return {email.split("@")[0] for email in emails}


def external_domains(event: CalendarEvent) -> set[str]:
    """Company part of non-firm email domains: "david@meridiancap.com" -> "meridiancap"."""
    domains = {email.split("@")[1] for email in event.attendees}
    return {domain.split(".")[0] for domain in domains if domain != FIRM_DOMAIN}


def is_virtual(location: str | None) -> bool | None:
    if not location:
        return None
    return any(keyword in location.lower() for keyword in VIRTUAL_KEYWORDS)


def person_attending(name: str | None, people: set[str]) -> float | None:
    if not name or not people:
        return None
    return float(email_local_part(name) in people)


def company_mentioned(company: str | None, event: CalendarEvent) -> float | None:
    distinctive = {word for word in tokens(company) if word not in GENERIC_COMPANY_WORDS and len(word) >= 3}
    if not distinctive:
        return None
    in_title = bool(distinctive & tokens(event.title))
    in_domain = any(word in domain for word in distinctive for domain in external_domains(event))
    return float(in_title or in_domain)


def compute_signals(crm: CrmRecord, event: CalendarEvent) -> dict[str, float | None]:
    signals: dict[str, float | None] = {}

    signals["same_date"] = (
        float(crm.meeting_date == event.meeting_date) if crm.meeting_date and event.meeting_date else None
    )
    if crm.start and event.start:
        gap_minutes = abs((crm.start - event.start).total_seconds()) / 60
        signals["time_proximity"] = max(0.0, 1 - gap_minutes / TIME_DECAY_MINUTES)
    else:
        signals["time_proximity"] = None

    people = calendar_people(event)
    signals["client_attending"] = person_attending(crm.client_name, people)
    signals["owner_attending"] = person_attending(crm.owner, people)
    signals["company_mentioned"] = company_mentioned(crm.client_company, event)
    signals["title_similarity"] = overlap(tokens(crm.subject), tokens(event.title))
    signals["location_similarity"] = overlap(tokens(crm.location), tokens(event.location))
    return signals


def confidence(signals: dict[str, float | None], weights: dict[str, float]) -> float:
    """Weighted average of the signals that have data."""
    available = {name: value for name, value in signals.items() if value is not None and weights[name] > 0}
    total_weight = sum(weights[name] for name in available)
    if total_weight == 0:
        return 0.0
    return sum(weights[name] * value for name, value in available.items()) / total_weight


def conflict_flags(crm: CrmRecord, event: CalendarEvent) -> list[str]:
    """Disagreements worth showing to a reviewer. They do not veto a match."""
    flags = []
    calendar_virtual = is_virtual(event.location)
    if crm.meeting_type in ("In-Person", "Virtual") and calendar_virtual is not None:
        if (crm.meeting_type == "Virtual") != calendar_virtual:
            flags.append(f"meeting_type_conflict: CRM {crm.meeting_type!r} vs calendar location {event.location!r}")
    if crm.start and event.start and crm.start != event.start:
        gap_minutes = round((event.start - crm.start).total_seconds() / 60)
        flags.append(f"start_time_differs_by_{gap_minutes}_min")
    if (crm.status or "").lower() == "cancelled" and (event.status or "").lower() != "cancelled":
        flags.append(f"status_conflict: CRM cancelled, calendar {event.status!r}")
    return flags
