"""REST API serving match predictions.

Run: uvicorn api:app --reload
The matcher runs once at startup; the data is small enough to keep in memory.
"""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from matching.match import run_matching, to_json

app = FastAPI(title="Record Matching API")
result = run_matching()
summary = to_json(result)
crm_ids = {record.id for record in result.crm_records}
calendar_ids = {event.id for event in result.calendar_events}


def candidates_for(field: str, record_id: str) -> list[dict]:
    pairs = [p for p in result.candidates if getattr(p, field) == record_id]
    return [asdict(p) for p in sorted(pairs, key=lambda p: p.confidence, reverse=True)]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "crm_records": len(crm_ids), "calendar_events": len(calendar_ids)}


@app.get("/matches")
def list_matches(decision: str | None = None) -> list[dict]:
    """All assigned pairs, optionally filtered by decision ("match" or "review")."""
    return [m for m in summary["matches"] if decision is None or m["decision"] == decision]


@app.get("/matches/crm/{crm_id}")
def match_for_crm(crm_id: str) -> dict:
    if crm_id not in crm_ids:
        raise HTTPException(status_code=404, detail=f"Unknown CRM id {crm_id}")
    return {
        "crm_id": crm_id,
        "matches": [m for m in summary["matches"] if m["crm_id"] == crm_id],
        "candidates": candidates_for("crm_id", crm_id),
        "data_quality_issues": summary["data_quality_issues"].get(crm_id, []),
    }


@app.get("/matches/calendar/{event_id}")
def match_for_calendar(event_id: str) -> dict:
    if event_id not in calendar_ids:
        raise HTTPException(status_code=404, detail=f"Unknown calendar id {event_id}")
    return {
        "calendar_id": event_id,
        "matches": [m for m in summary["matches"] if m["calendar_id"] == event_id],
        "candidates": candidates_for("calendar_id", event_id),
        "data_quality_issues": summary["data_quality_issues"].get(event_id, []),
    }


@app.get("/duplicates")
def duplicates() -> list[list[str]]:
    return summary["calendar_duplicate_clusters"]
