"""Run the whole pipeline: ingest -> match -> evaluate -> write outputs.

Usage: python run_pipeline.py          (set LOG_LEVEL=DEBUG for per-pair detail)
"""

import json
import logging
import time

from matching import evaluate
from matching.config import DATA_DIR, OUTPUT_DIR, WEIGHTS, setup_logging
from matching.match import run_matching, to_json
from matching.ml_experiment import run_experiment
from matching.report import render

logger = logging.getLogger("pipeline")


def main() -> None:
    started = time.perf_counter()
    result = run_matching()
    logger.info("Ingested %d CRM records and %d calendar events",
                len(result.crm_records), len(result.calendar_events))
    for record in [*result.crm_records, *result.calendar_events]:
        if record.issues:
            logger.warning("Data quality %s: %s", record.id, "; ".join(record.issues))

    decisions = [p.decision for p in result.matches]
    logger.info("Matching: %d candidate pairs scored, %d matches, %d for review, %d duplicate cluster(s)",
                len(result.candidates), decisions.count("match"), decisions.count("review"),
                sum(1 for c in result.duplicate_clusters if len(c) > 1))

    labels = evaluate.load_labels(
        DATA_DIR / "evaluation_labels.json",
        [record.id for record in result.crm_records],
        [event.id for event in result.calendar_events],
    )

    rules = evaluate.predicted_matches(result)
    baseline = evaluate.baseline_matches(result)
    labeled_pairs = labels.combined.keys()
    results = {
        "labels": labels,
        "weights": WEIGHTS,
        "rules": {"explicit": evaluate.metrics(rules, labels.explicit),
                  "combined": evaluate.metrics(rules, labels.combined)},
        "baseline": {"explicit": evaluate.metrics(baseline, labels.explicit),
                     "combined": evaluate.metrics(baseline, labels.combined)},
        "rules_errors": evaluate.errors(rules, labels.combined),
        "baseline_errors": evaluate.errors(baseline, labels.combined),
        "margins": evaluate.labeled_pair_scores(result, labels.explicit),
        "sweep": evaluate.threshold_sweep(result, labels),
        "ablation": evaluate.ablation(result, labels),
        "ml": run_experiment(result, labels.explicit, labels.combined),
        "unlabeled": [p for p in result.matches if (p.crm_id, p.calendar_id) not in labeled_pairs],
        "near_misses": [p for p in result.candidates if p.decision == "no_match" and p.confidence >= 0.4],
        "duplicates": evaluate.duplicate_check(result, labels),
        "issues": to_json(result)["data_quality_issues"],
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "matches.json", "w", encoding="utf-8") as file:
        json.dump(to_json(result), file, indent=2)
    with open(OUTPUT_DIR / "evaluation.md", "w", encoding="utf-8") as file:
        file.write(render(results))

    for name, scores in [("baseline", results["baseline"]["explicit"]),
                         ("rules", results["rules"]["explicit"]),
                         ("logistic regression (LOO)", results["ml"]["explicit_loo"])]:
        logger.info("Evaluation %s on explicit labels: precision=%.2f recall=%.2f f1=%.2f",
                    name, scores["precision"], scores["recall"], scores["f1"])
    logger.info("%d predicted matches have no label and need manual review", len(results["unlabeled"]))
    logger.info("Wrote %s and %s in %.2fs", OUTPUT_DIR / "matches.json", OUTPUT_DIR / "evaluation.md",
                time.perf_counter() - started)


if __name__ == "__main__":
    setup_logging()
    main()
