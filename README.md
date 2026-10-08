# Record Matching: CRM ↔ Calendar

Decides which CRM meeting records and calendar events describe the same real-world meeting,
evaluates the decisions against the partial labels, and serves them over a REST API.

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync                               # create .venv and install the locked dependencies
uv run python run_pipeline.py         # ingest -> match -> evaluate; writes output/matches.json and output/evaluation.md
uv run uvicorn api:app --reload       # API on http://127.0.0.1:8000 (interactive docs at /docs)
uv run pytest                         # 19 tests
```

If `uv sync` fails with a hardlink error (common in OneDrive or other synced folders), use
`uv sync --link-mode=copy`, or set `UV_LINK_MODE=copy`.

| Endpoint | Returns |
|---|---|
| `GET /health` | Record counts |
| `GET /matches?decision=match\|review` | All assigned pairs with confidence, signal breakdown and flags |
| `GET /matches/crm/{crm_id}` | That record's match, every scored candidate, and its data-quality issues |
| `GET /matches/calendar/{event_id}` | Same, from the calendar side |
| `GET /duplicates` | Calendar events detected as the same meeting |

## Approach

```
ingest.py      load + normalize both sources, record data-quality issues per record
dedupe.py      group calendar events that are the same meeting (CAL-A5 / CAL-A6)
features.py    compare one CRM record with one calendar event -> 7 signals in [0, 1]
match.py       block on date, score pairs, assign one-to-one, decide match / review / no_match
evaluate.py    metrics, baseline, threshold sweep, ablation, margins
ml_experiment.py  logistic regression on the same signals, cross-validated
report.py      render output/evaluation.md
```

**Scoring.** Each candidate pair gets seven signals:

| Signal | Weight | Meaning |
|---|---|---|
| `same_date` | 3.0 | Same calendar day |
| `time_proximity` | 2.0 | 1 at the same start time, falling linearly to 0 at 4 hours apart |
| `client_attending` | 3.0 | CRM client's likely email (`first.last`) is an attendee |
| `owner_attending` | 1.0 | Relationship owner is an attendee or organizer |
| `company_mentioned` | 1.5 | A distinctive company word appears in the title or an attendee's email domain |
| `title_similarity` | 1.0 | Share of the shorter title's words found in the other |
| `location_similarity` | 0.5 | Same, for locations |

Confidence is the weighted average of the signals **that have data**. A missing field
(e.g. CRM-1007 has no time) is skipped rather than scored as a mismatch. Each pair is then
classified: **match** (confidence ≥ 0.70), **review** (≥ 0.50) or **no_match**.

**Assignment.** Pairs are assigned greedily, highest confidence first, so each CRM record matches at most one
meeting and each meeting at most one CRM record. Duplicate calendar events form one meeting, so
both CAL-A5 and CAL-A6 match CRM-1005.

## Results

Full report: [output/evaluation.md](output/evaluation.md). Summary:

| Method | Precision | Recall | F1 (13 explicit labels) |
|---|---|---|---|
| Naive baseline: same date + client attending | 1.00 | 0.75 | 0.86 |
| **Rules matcher (submitted)** | **1.00** | **1.00** | **1.00** |
| Logistic regression, leave-one-out CV | 1.00 | 1.00 | 1.00 |

The same numbers hold on the larger set with 261 implied negatives (explained below).
**A perfect score on these labels is weak evidence**, and the report says so:

- **The labels are easy.** All 5 labeled non-matches are on a different date from the CRM record,
  so date blocking rejects them before scoring. To check the scoring itself, I scored the labeled
  pairs with blocking turned off. The lowest labeled match scored **0.917** and the highest labeled
  non-match scored **0.458**, so the score separates them with room to spare.
- **The baseline shows what the extra signals buy.** The baseline misses CRM-1006/CAL-A7 (an internal
  meeting with no client) and CRM-1017/CAL-A20 (client name is the placeholder "Multiple").
- **Threshold sweep.** Results are unchanged for any match threshold from 0.50 to 0.85.
- **Ablation.** Removing any single signal changes none of the 18 predictions. The signals are
  redundant enough to survive losing one, but it also means this data cannot pin down exact weights.
- **Unlabeled predictions.** 10 of the 18 predicted pairs have no label. They are listed in the report
  for manual review and not counted as correct. On inspection they look right, but that is my
  judgment, not ground truth.

### Rules vs. machine learning

Logistic regression on the same seven signals was evaluated with leave-one-out cross-validation
(on the 13 explicit labels) and with leave-one-CRM-record-out (on explicit + implied labels, so pairs
sharing a record never sit on both sides of a split). It matched the rules exactly. Fitted on all
labels, it puts the most weight on date and time, which agrees with the hand-set weights.

**Decision: keep the rules matcher.** The ML model is not better on this data, needs labels to
train, and is harder to explain. With a few hundred labeled pairs that include hard negatives
(same client, same day, different meeting), I would revisit this.

## Key decisions

| Decision | Chosen | Considered | Why |
|---|---|---|---|
| Method | Weighted rules | Trained classifier, embeddings, LLM, Fellegi–Sunter | 13 labels cannot train or validate a model; rules are explainable and keep the labels purely for evaluation. The ML experiment confirmed it adds nothing here. |
| Weights | Set by reasoning before the first evaluation run | Fitting to labels | Avoids tuning to the labels I evaluate on. They were not changed after seeing results. |
| Candidate blocking | Records within ±1 day | Exact date only; no blocking | Same date still decides most cases through the `same_date` signal; ±1 day protects against timezone shifts across midnight. |
| Time | Soft signal, decays over 4 hours | Hard window (e.g. ±30 min) | CRM-1016/CAL-A17 is a labeled match 2 hours apart; a hard window would reject it. |
| Timezone | `Z` timestamps converted to US/Eastern; naive timestamps assumed Eastern | Treat all as UTC; ignore offsets | Offices are NYC/DC/Boston. CAL-A4 (19:00Z) becomes 15:00, still an hour off CRM's 14:00, so time stays a soft signal. |
| Missing fields | Excluded from the confidence average | Score as 0 | A missing value is no evidence, not counter-evidence. |
| Conflicting fields (location type, status) | Flagged, do not veto | Penalize or reject | The CRM and calendar disagree on real matches (CRM-1002 says In-Person, the calendar says Zoom). |
| Duplicates | Cluster calendar events first, then match clusters | Strict one-to-one; ignore | Strict one-to-one would drop one of CAL-A5/A6. |
| Cancelled CRM-1009 vs CAL-A10 | Match, with `status_conflict` flag | Exclude cancelled meetings | They describe the same scheduled slot; the calendar is stale. A reviewer should see the conflict rather than lose the link. |
| Placeholders (`"Multiple"`, `"external-guests"`) | Treated as missing | Use as-is | `"Multiple"` is not a person; matching on it would be noise. |
| Bad data | Repair where unambiguous, flag everything | Drop bad records | Dropping CRM-1008 (bad date) would lose a labeled match. Every issue is listed in the output and the API. |
| Implied negatives | Derived from labeled matches, reported separately | Use explicit labels only | Gives 274 labeled pairs instead of 13. Relies on the assumption below. |

## Assumptions

- Naive timestamps are US/Eastern.
- Ambiguous dates are month-first (`03-15/2025` → 15 March).
- A meeting appears at most once per source, apart from labeled duplicates. Implied negatives depend on this.
- Calendar emails follow `first.last@company-domain`.
- `firma.com` is the firm's own domain.

## Where it fails

- **Rescheduled meetings.** Blocking on ±1 day means a meeting moved to another week is never compared.
- **Sparse records score on date alone.** CRM-1006 vs CAL-A11 (empty calendar entry) scores 0.57, which is
  review level, from same date plus nearby time only. It was not assigned because CAL-A7 scored higher.
- **Generic company words.** "Fund VII Investors" matches any title containing "VII", so CRM-1017 vs
  CAL-A19 scores 0.61. Again saved only by a better candidate.
- **Several meetings with one client on one day** would compete, and only time and title would separate them.
- **Name matching is exact** on `first.last`. Nicknames, initials or other email formats would be missed.
- **Recurring events** are treated as single events; the CRM-1006/CAL-A3 label shows instances must not
  cross-match, which works here only because the dates differ.

## With more time

- Get the firm's real timezone and more labels, especially hard negatives on the same day.
- Expand recurring events into occurrences; allow cross-date matching for meetings with a reschedule note.
- Fuzzy name matching (nicknames, initials) and per-domain email patterns.
- Persist predictions and reviewer decisions so the review queue feeds back into labels.

## Time spent
3hrs

## Use of AI tools

- I used Claude Code to explore the brief and data, discuss the approach, and generate the code, tests
  and this README. All results reported here come from actually running the pipeline and tests.
- Corrections during the session: one unit test had a wrong expected value (it did not account for
  "discussion" being a stopword), and the report's headline table was malformed. Both were caught by
  running the tests and reading the generated report.

