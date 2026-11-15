"""Two contamination checks for public test data.

1. Continuation test: give the first half of a test text and measure how much of
   the true second half the model reproduces verbatim.
2. Perturbation test: mask the most label-discriminative words (learned from the
   training set) and compare accuracy on masked test vs masked validation texts.
   A model that keeps much higher accuracy on masked TEST texts than on masked
   VALIDATION texts may have memorised the test set.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2


def continuation_prompt(prefix):
    return [{"role": "user", "content":
             "Continue the following Indonesian text exactly as it originally appeared. "
             "Output only the continuation.\n\n" + prefix}]


def split_half(text):
    toks = text.split()
    h = len(toks) // 2
    return " ".join(toks[:h]), " ".join(toks[h:])


def overlap_ratio(reference, generated, n_first=10):
    """Share of the first n reference tokens reproduced in order at the start of the output."""
    ref = reference.lower().split()[:n_first]
    gen = re.sub(r"\s+", " ", generated.lower()).split()[:n_first]
    hits = 0
    for r, g in zip(ref, gen):
        if r != g:
            break
        hits += 1
    return hits / max(1, len(ref))


def discriminative_terms(train_texts, train_labels, k=200):
    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", min_df=3)
    X = vec.fit_transform(train_texts)
    scores, _ = chi2(X, train_labels)
    vocab = np.array(vec.get_feature_names_out())
    return set(vocab[np.argsort(np.nan_to_num(scores))[::-1][:k]])


def mask_text(text, terms, token="[...]"):
    return " ".join(token if w.lower().strip(".,!?\"'") in terms else w for w in text.split())
