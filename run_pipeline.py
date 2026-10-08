"""Run the whole pipeline: ingest -> match -> evaluate -> write outputs.

Usage: python run_pipeline.py
"""

import json

from matching import evaluate
from matching.config import DATA_DIR, OUTPUT_DIR, WEIGHTS
from matching.match import run_matching, to_json
from matching.ml_experiment import run_experiment
from matching.report import render


def main() -> None:
    result = run_matching()
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

    decisions = [p.decision for p in result.matches]
    print(f"Matches: {decisions.count('match')}, needs review: {decisions.count('review')}")
    for name in ("baseline", "rules"):
        m = results[name]["explicit"]
        print(f"{name:>8} on explicit labels: precision={m['precision']:.2f} recall={m['recall']:.2f} f1={m['f1']:.2f}")
    print(f"Wrote {OUTPUT_DIR / 'matches.json'} and {OUTPUT_DIR / 'evaluation.md'}")


if __name__ == "__main__":
    main()
