"""Score candidate pairs and assign each CRM record to at most one calendar meeting."""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from matching.config import CANDIDATE_WINDOW_DAYS, DATA_DIR, MATCH_THRESHOLD, REVIEW_THRESHOLD, WEIGHTS
from matching.dedupe import find_duplicate_clusters
from matching.features import compute_signals, confidence, conflict_flags
from matching.ingest import CalendarEvent, CrmRecord, load_calendar, load_crm

logger = logging.getLogger(__name__)


@dataclass
class PairScore:
    crm_id: str
    calendar_id: str
    confidence: float
    signals: dict[str, float | None]
    flags: list[str]
    decision: str = "no_match"


@dataclass
class MatchingResult:
    crm_records: list[CrmRecord]
    calendar_events: list[CalendarEvent]
    duplicate_clusters: list[list[str]]
    candidates: list[PairScore]
    matches: list[PairScore] = field(default_factory=list)


def is_candidate(crm: CrmRecord, event: CalendarEvent) -> bool:
    """Only compare records within a day of each other (blocking)."""
    if crm.meeting_date is None or event.meeting_date is None:
        return False
    return abs((crm.meeting_date - event.meeting_date).days) <= CANDIDATE_WINDOW_DAYS


def score_pair(crm: CrmRecord, event: CalendarEvent, weights: dict[str, float]) -> PairScore:
    signals = compute_signals(crm, event)
    return PairScore(
        crm_id=crm.id,
        calendar_id=event.id,
        confidence=round(confidence(signals, weights), 3),
        signals={name: None if value is None else round(value, 3) for name, value in signals.items()},
        flags=conflict_flags(crm, event),
    )


def decide(score: float) -> str:
    if score >= MATCH_THRESHOLD:
        return "match"
    if score >= REVIEW_THRESHOLD:
        return "review"
    return "no_match"


def assign(candidates: list[PairScore], clusters: list[list[str]]) -> list[PairScore]:
    """Greedy one-to-one assignment between CRM records and calendar clusters.

    A cluster of duplicate events counts as one meeting, so every event in it can
    match the same CRM record. Highest-confidence pairs are assigned first.
    """
    cluster_of = {event_id: index for index, cluster in enumerate(clusters) for event_id in cluster}
    by_confidence = sorted(candidates, key=lambda pair: pair.confidence, reverse=True)

    taken_crm: set[str] = set()
    taken_clusters: set[int] = set()
    matches = []
    for pair in by_confidence:
        if pair.confidence < REVIEW_THRESHOLD:
            break
        cluster = cluster_of[pair.calendar_id]
        if pair.crm_id in taken_crm or cluster in taken_clusters:
            continue
        taken_crm.add(pair.crm_id)
        taken_clusters.add(cluster)
        siblings = [p for p in candidates if p.crm_id == pair.crm_id and cluster_of[p.calendar_id] == cluster]
        for sibling in siblings:
            if sibling.confidence >= REVIEW_THRESHOLD:
                sibling.decision = decide(sibling.confidence)
                if len(clusters[cluster]) > 1:
                    sibling.flags.append(f"duplicate_calendar_cluster: {clusters[cluster]}")
                matches.append(sibling)
    return sorted(matches, key=lambda pair: (pair.crm_id, pair.calendar_id))


def run_matching(weights: dict[str, float] = WEIGHTS, data_dir: Path = DATA_DIR) -> MatchingResult:
    crm_records = load_crm(data_dir / "crm_events.json")
    calendar_events = load_calendar(data_dir / "calendar_events.json")
    clusters = [[event.id for event in cluster] for cluster in find_duplicate_clusters(calendar_events)]

    candidates = [
        score_pair(crm, event, weights)
        for crm in crm_records
        for event in calendar_events
        if is_candidate(crm, event)
    ]
    result = MatchingResult(crm_records, calendar_events, clusters, candidates)
    result.matches = assign(candidates, clusters)

    logger.debug("Duplicate calendar clusters: %s", [c for c in clusters if len(c) > 1])
    logger.debug("Scored %d candidate pairs, assigned %d", len(candidates), len(result.matches))
    for pair in result.matches:
        logger.debug("%s -> %s confidence=%.3f decision=%s flags=%s",
                     pair.crm_id, pair.calendar_id, pair.confidence, pair.decision, pair.flags)
    return result


def to_json(result: MatchingResult) -> dict:
    matched_crm = {pair.crm_id for pair in result.matches}
    matched_calendar = {pair.calendar_id for pair in result.matches}
    return {
        "matches": [asdict(pair) for pair in result.matches],
        "unmatched_crm": [r.id for r in result.crm_records if r.id not in matched_crm],
        "unmatched_calendar": [e.id for e in result.calendar_events if e.id not in matched_calendar],
        "calendar_duplicate_clusters": [c for c in result.duplicate_clusters if len(c) > 1],
        "candidates": [asdict(pair) for pair in result.candidates],
        "data_quality_issues": {
            record.id: record.issues
            for record in [*result.crm_records, *result.calendar_events]
            if record.issues
        },
    }
