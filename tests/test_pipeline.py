"""End-to-end checks on the real data files."""

import pytest
from fastapi.testclient import TestClient

from api import app
from matching import evaluate
from matching.config import DATA_DIR
from matching.match import run_matching


@pytest.fixture(scope="module")
def result():
    return run_matching()


@pytest.fixture(scope="module")
def labels(result):
    return evaluate.load_labels(
        DATA_DIR / "evaluation_labels.json",
        [r.id for r in result.crm_records],
        [e.id for e in result.calendar_events],
    )


def decision(result, crm_id, calendar_id):
    return next((p.decision for p in result.matches if (p.crm_id, p.calendar_id) == (crm_id, calendar_id)), None)


def test_explicit_labels_are_reproduced(result, labels):
    predicted = evaluate.predicted_matches(result)
    for pair, is_match in labels.explicit.items():
        assert (pair in predicted) == is_match, pair


def test_each_crm_record_matches_at_most_one_meeting(result):
    clusters = {e: tuple(c) for c in result.duplicate_clusters for e in c}
    meetings_per_crm = {}
    for pair in result.matches:
        meetings_per_crm.setdefault(pair.crm_id, set()).add(clusters[pair.calendar_id])
    assert all(len(meetings) == 1 for meetings in meetings_per_crm.values())


def test_calendar_duplicates_both_map_to_same_crm_record(result):
    assert ["CAL-A5", "CAL-A6"] in result.duplicate_clusters
    assert decision(result, "CRM-1005", "CAL-A5") == "match"
    assert decision(result, "CRM-1005", "CAL-A6") == "match"


def test_bad_data_records_still_match(result):
    assert decision(result, "CRM-1007", "CAL-A8") == "match"   # missing time
    assert decision(result, "CRM-1008", "CAL-A9") == "match"   # malformed date
    assert decision(result, "CRM-1004", "CAL-A4") == "match"   # UTC timestamp


def test_conflicts_are_flagged_not_rejected(result):
    pair = next(p for p in result.matches if (p.crm_id, p.calendar_id) == ("CRM-1002", "CAL-A2"))
    assert pair.decision == "match"
    assert any(flag.startswith("meeting_type_conflict") for flag in pair.flags)


def test_implied_negatives_exclude_labeled_duplicate(labels):
    assert ("CRM-1001", "CAL-A2") in labels.implied
    assert all(not is_match for is_match in labels.implied.values())
    assert not set(labels.implied) & set(labels.explicit)


def test_api_endpoints():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/matches/crm/CRM-1001").json()
    assert body["matches"][0]["calendar_id"] == "CAL-A1"
    assert client.get("/matches/calendar/CAL-A3").json()["matches"] == []
    assert client.get("/matches/crm/CRM-9999").status_code == 404
    assert all(m["decision"] == "match" for m in client.get("/matches?decision=match").json())
