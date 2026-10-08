"""Find calendar events that describe the same meeting (e.g. a re-sent invite)."""

from matching.config import DUPLICATE_MAX_GAP_MINUTES, FIRM_DOMAIN
from matching.ingest import CalendarEvent


def external_attendees(event: CalendarEvent) -> set[str]:
    return {email for email in event.attendees if not email.endswith("@" + FIRM_DOMAIN)}


def are_duplicates(a: CalendarEvent, b: CalendarEvent) -> bool:
    """Same organizer, same day, close start times and at least one shared client attendee.

    Requiring a shared external attendee keeps back-to-back internal meetings apart.
    """
    if not (a.start and b.start) or a.organizer != b.organizer:
        return False
    gap_minutes = abs((a.start - b.start).total_seconds()) / 60
    return (
        a.start.date() == b.start.date()
        and gap_minutes <= DUPLICATE_MAX_GAP_MINUTES
        and bool(external_attendees(a) & external_attendees(b))
    )


def find_duplicate_clusters(events: list[CalendarEvent]) -> list[list[CalendarEvent]]:
    """Group events into clusters; most clusters hold a single event."""
    clusters: list[list[CalendarEvent]] = []
    for event in events:
        home = next((cluster for cluster in clusters if any(are_duplicates(event, other) for other in cluster)), None)
        if home is None:
            clusters.append([event])
        else:
            home.append(event)
    return clusters
