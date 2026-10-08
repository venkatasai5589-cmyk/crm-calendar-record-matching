"""Measure the matcher against the partial labels.

Two label sets are used:
- explicit: the 13 labeled cross-source pairs, exactly as provided.
- explicit + implied: adds negatives derived from the labeled matches. If CRM-1001
  matches CAL-A1, then CRM-1001 matches no other calendar event (except a labeled
  duplicate of CAL-A1), and CAL-A1 matches no other CRM record. This assumes each
  meeting appears once per source, apart from labeled duplicates.

Predicted matches that no label covers cannot be scored; they are listed for review.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from matching.config import MATCH_THRESHOLD, WEIGHTS
from matching.features import calendar_people, email_local_part
from matching.match import MatchingResult, run_matching, score_pair

Pair = tuple[str, str]


@dataclass
class Labels:
    explicit: dict[Pair, bool]
    implied: dict[Pair, bool]
    calendar_duplicates: list[Pair]

    @property
    def combined(self) -> dict[Pair, bool]:
        return {**self.implied, **self.explicit}


def load_labels(path: Path, crm_ids: list[str], calendar_ids: list[str]) -> Labels:
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    explicit = {(row["crm_id"], row["calendar_id"]): row["match"] for row in raw["cross_source_pairs"]}
    duplicates = [
        (row["record_a"], row["record_b"])
        for row in raw["intra_source_duplicates"]
        if row["source"] == "calendar" and row["duplicate"]
    ]
    return Labels(explicit, implied_negatives(explicit, crm_ids, calendar_ids, duplicates), duplicates)


def implied_negatives(
    explicit: dict[Pair, bool], crm_ids: list[str], calendar_ids: list[str], duplicates: list[Pair]
) -> dict[Pair, bool]:
    same_meeting = {a: {a} for a in calendar_ids}
    for a, b in duplicates:
        same_meeting[a].add(b)
        same_meeting[b].add(a)

    implied = {}
    for (crm_id, calendar_id), is_match in explicit.items():
        if not is_match:
            continue
        for other_calendar in calendar_ids:
            if other_calendar not in same_meeting[calendar_id]:
                implied[(crm_id, other_calendar)] = False
        for other_crm in crm_ids:
            if other_crm != crm_id:
                for event_id in same_meeting[calendar_id]:
                    implied[(other_crm, event_id)] = False
    return {pair: value for pair, value in implied.items() if pair not in explicit}


def metrics(predicted: set[Pair], labels: dict[Pair, bool]) -> dict:
    tp = sum(1 for pair, is_match in labels.items() if is_match and pair in predicted)
    fp = sum(1 for pair, is_match in labels.items() if not is_match and pair in predicted)
    fn = sum(1 for pair, is_match in labels.items() if is_match and pair not in predicted)
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def predicted_matches(result: MatchingResult, threshold: float = MATCH_THRESHOLD) -> set[Pair]:
    return {(pair.crm_id, pair.calendar_id) for pair in result.matches if pair.confidence >= threshold}


def baseline_matches(result: MatchingResult) -> set[Pair]:
    """Naive matcher: same date and the CRM client appears among the calendar attendees."""
    return {
        (crm.id, event.id)
        for crm in result.crm_records
        for event in result.calendar_events
        if crm.meeting_date
        and crm.meeting_date == event.meeting_date
        and crm.client_name
        and email_local_part(crm.client_name) in calendar_people(event)
    }


def threshold_sweep(result: MatchingResult, labels: Labels) -> list[dict]:
    rows = []
    for step in range(10):
        threshold = round(0.50 + step * 0.05, 2)
        predicted = predicted_matches(result, threshold)
        rows.append({
            "threshold": threshold,
            "explicit": metrics(predicted, labels.explicit),
            "combined": metrics(predicted, labels.combined),
            "predicted": len(predicted),
        })
    return rows


def ablation(full_result: MatchingResult, labels: Labels) -> list[dict]:
    """Re-run the whole pipeline with one signal switched off at a time.

    "changed" counts predictions that differ from the full model, which also covers
    pairs the labels say nothing about.
    """
    full_predictions = predicted_matches(full_result)
    rows = []
    for signal in WEIGHTS:
        predicted = predicted_matches(run_matching({**WEIGHTS, signal: 0.0}))
        rows.append({
            "removed": signal,
            "explicit": metrics(predicted, labels.explicit),
            "combined": metrics(predicted, labels.combined),
            "changed": sorted(predicted ^ full_predictions),
        })
    return rows


def labeled_pair_scores(result: MatchingResult, labels: dict[Pair, bool]) -> list[dict]:
    """Score every labeled pair directly, skipping the date-window blocking.

    The labeled negatives all fall on different dates, so blocking alone rejects them.
    This shows whether the scoring would also separate them.
    """
    crm_by_id = {record.id: record for record in result.crm_records}
    event_by_id = {event.id: event for event in result.calendar_events}
    rows = []
    for (crm_id, calendar_id), is_match in sorted(labels.items(), key=lambda item: not item[1]):
        pair = score_pair(crm_by_id[crm_id], event_by_id[calendar_id], WEIGHTS)
        rows.append({"pair": (crm_id, calendar_id), "label": is_match, "confidence": pair.confidence})
    return rows


def errors(predicted: set[Pair], labels: dict[Pair, bool]) -> dict[str, list[Pair]]:
    return {
        "false_positives": sorted(p for p, is_match in labels.items() if not is_match and p in predicted),
        "false_negatives": sorted(p for p, is_match in labels.items() if is_match and p not in predicted),
    }


def duplicate_check(result: MatchingResult, labels: Labels) -> dict:
    predicted = {
        tuple(sorted((a, b)))
        for cluster in result.duplicate_clusters
        for i, a in enumerate(cluster)
        for b in cluster[i + 1:]
    }
    labeled = {tuple(sorted(pair)) for pair in labels.calendar_duplicates}
    return {"labeled": sorted(labeled), "predicted": sorted(predicted), "found": sorted(labeled & predicted)}
