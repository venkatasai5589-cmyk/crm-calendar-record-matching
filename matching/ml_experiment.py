"""Experiment: would a learned model beat the hand-set weights?

Logistic regression on the same signals the rules use. Every prediction is made
by a model that never saw that pair (or that CRM record) during training, so the
scores are comparable to the rules matcher, which uses no labels at all.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict

from matching.config import WEIGHTS
from matching.evaluate import Pair, metrics
from matching.features import compute_signals
from matching.match import MatchingResult

FEATURES = list(WEIGHTS)
# A missing signal is fed to the model as 0.5, i.e. "no evidence either way".
MISSING_VALUE = 0.5


def feature_matrix(result: MatchingResult, pairs: list[Pair]) -> np.ndarray:
    crm_by_id = {record.id: record for record in result.crm_records}
    event_by_id = {event.id: event for event in result.calendar_events}
    rows = []
    for crm_id, calendar_id in pairs:
        signals = compute_signals(crm_by_id[crm_id], event_by_id[calendar_id])
        rows.append([MISSING_VALUE if signals[name] is None else signals[name] for name in FEATURES])
    return np.array(rows)


def new_model() -> LogisticRegression:
    # Balanced class weights: implied negatives far outnumber the positives.
    return LogisticRegression(class_weight="balanced", max_iter=1000)


def cross_validated(result: MatchingResult, labels: dict[Pair, bool], group_by_crm: bool) -> dict:
    """Leave-one-out over pairs, or over CRM records so no record's pairs leak into training."""
    pairs = sorted(labels)
    X = feature_matrix(result, pairs)
    y = np.array([labels[pair] for pair in pairs], dtype=int)
    if group_by_crm:
        cv, groups = LeaveOneGroupOut(), [crm_id for crm_id, _ in pairs]
    else:
        cv, groups = LeaveOneOut(), None
    predictions = cross_val_predict(new_model(), X, y, cv=cv, groups=groups)
    predicted = {pair for pair, label in zip(pairs, predictions) if label == 1}
    return metrics(predicted, labels)


def learned_weights(result: MatchingResult, labels: dict[Pair, bool]) -> dict[str, float]:
    pairs = sorted(labels)
    model = new_model().fit(feature_matrix(result, pairs), [labels[pair] for pair in pairs])
    return {name: round(float(coef), 2) for name, coef in zip(FEATURES, model.coef_[0])}


def run_experiment(result: MatchingResult, explicit: dict[Pair, bool], combined: dict[Pair, bool]) -> dict:
    return {
        "explicit_loo": cross_validated(result, explicit, group_by_crm=False),
        "combined_grouped": cross_validated(result, combined, group_by_crm=True),
        "weights_explicit": learned_weights(result, explicit),
        "weights_combined": learned_weights(result, combined),
    }
