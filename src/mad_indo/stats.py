"""Metrics and paired significance tests."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def metrics(gold, pred, labels):
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    out = {"accuracy": accuracy_score(gold, pred),
           "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
           "unknown_rate": float(np.mean(np.asarray(pred) == "unknown"))}
    for lab, pi, ri, fi in zip(labels, p, r, f):
        out[f"f1_{lab}"] = fi
        out[f"recall_{lab}"] = ri
    return out


def majority_across_runs(pred_matrix):
    """pred_matrix: list of equal-length prediction lists (one per seed/paraphrase).
    Returns one prediction per sample; ties go to the first run's label among the tied."""
    out = []
    for preds in zip(*pred_matrix):
        c = Counter(preds)
        top = max(c.values())
        out.append(next(p for p in preds if c[p] == top))
    return out


def mcnemar_exact(gold, pred_a, pred_b):
    """Exact McNemar test on paired correctness. Returns (b, c, p): b = A right & B wrong."""
    g, a, b_ = map(np.asarray, (gold, pred_a, pred_b))
    ca, cb = a == g, b_ == g
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def paired_bootstrap_diff(gold, pred_a, pred_b, labels, n=1000, seed=2026):
    """95% CI of macro-F1(A) - macro-F1(B), resampling the same samples for both."""
    rng = np.random.default_rng(seed)
    g, a, b = map(np.asarray, (gold, pred_a, pred_b))
    idx = np.arange(len(g))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(f1_score(g[s], a[s], labels=labels, average="macro", zero_division=0)
                     - f1_score(g[s], b[s], labels=labels, average="macro", zero_division=0))
    point = (f1_score(g, a, labels=labels, average="macro", zero_division=0)
             - f1_score(g, b, labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi
