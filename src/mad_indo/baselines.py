"""B1 TF-IDF baseline and B2 frozen-embedding classifier (shared selection logic)."""
from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC


def tfidf_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


def select_and_fit(make_model, C_grid, Xtr, ytr, Xva, yva, labels):
    """Pick C on validation macro-F1, return fitted model and a selection log."""
    log, best = [], (-1, None, None)
    for C in C_grid:
        t0 = time.perf_counter()
        m = make_model(C).fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), labels=labels, average="macro")
        log.append({"C": C, "valid_macro_f1": f1, "fit_s": time.perf_counter() - t0})
        if f1 > best[0]:
            best = (f1, C, m)
    return best[2], best[1], log


def run_tfidf(train, valid, test, labels, C_grid):
    results = {}
    for name, ctor in {
        "B1_tfidf_logreg": lambda C: make_pipeline(tfidf_features(), LogisticRegression(C=C, max_iter=3000)),
        "B1_tfidf_linsvm": lambda C: make_pipeline(tfidf_features(), LinearSVC(C=C)),
    }.items():
        t0 = time.perf_counter()
        model, C, log = select_and_fit(ctor, C_grid, train.text, train.label, valid.text, valid.label, labels)
        t1 = time.perf_counter()
        pred = model.predict(test.text)
        infer_s = time.perf_counter() - t1
        results[name] = {"pred": list(pred), "best_C": C, "selection": log,
                         "train_select_s": t1 - t0, "infer_s_per_1000": 1000 * infer_s / len(test)}
    return results


def run_frozen(  # dual=False: primal solver, ~25x faster when n_samples > n_features (dense embeddings)
        Etr, ytr, Eva, yva, Ete, labels, C_grid):
    t0 = time.perf_counter()
    model, C, log = select_and_fit(lambda C: LinearSVC(C=C, dual=False, max_iter=5000), C_grid, Etr, ytr, Eva, yva, labels)
    t1 = time.perf_counter()
    pred = model.predict(Ete)
    return {"pred": list(pred), "best_C": C, "selection": log, "train_select_s": t1 - t0,
            "classifier_infer_s_per_1000": 1000 * (time.perf_counter() - t1) / len(Ete)}
