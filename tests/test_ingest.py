from datetime import date, datetime

from matching.ingest import clean_email, clean_name, parse_calendar_event, parse_crm_record, parse_date


def crm_row(**overrides):
    row = {
        "crm_id": "CRM-1", "subject": "Review", "client_name": "David Park", "client_company": "Meridian Capital",
        "relationship_owner": "Sarah Chen", "meeting_date": "2025-03-10", "meeting_time": "14:00",
        "meeting_type": "In-Person", "location": None, "status": "Confirmed",
    }
    return {**row, **overrides}


def test_malformed_date_is_parsed_and_flagged():
    issues = []
    assert parse_date("03-15/2025", issues) == date(2025, 3, 15)
    assert issues == ["nonstandard_date_format: '03-15/2025'"]


def test_unparseable_date_becomes_missing():
    issues = []
    assert parse_date("next tuesday", issues) is None
    assert issues[0].startswith("unparseable_date")


def test_missing_time_keeps_the_date():
    record = parse_crm_record(crm_row(meeting_time=None))
    assert record.meeting_date == date(2025, 3, 10)
    assert record.start is None
    assert "missing_time" in record.issues


def test_utc_timestamp_is_converted_to_eastern():
    event = parse_calendar_event({
        "event_id": "CAL-1", "title": "DD", "organizer": "sarah.chen@firma.com", "attendees": ["a@b.com"],
        "start_time": "2025-03-13T19:00:00Z", "end_time": "2025-03-13T20:30:00Z",
    })
    # 13 March 2025 is after the US DST change, so UTC-4.
    assert event.start == datetime(2025, 3, 13, 15, 0)
    assert any("converted_from_utc" in issue for issue in event.issues)


def test_obfuscated_email_is_repaired_and_junk_attendee_dropped():
    issues = []
    assert clean_email("raj.patel[at]atlasvc.com", issues) == "raj.patel@atlasvc.com"
    assert clean_email("external-guests", issues) is None
    assert len(issues) == 2


def test_placeholder_client_name_is_treated_as_missing():
    issues = []
    assert clean_name("Multiple", issues) is None
    assert issues == ["placeholder_client_name: 'Multiple'"]
