"""Apply manual audit corrections back to the result files.

Workflow: notebook 06 writes audit_manual_check.csv; the researcher fills in
`manual_label` where the parsed label is wrong and saves the file as
audit_manual_check_checked.csv. apply_corrections() then rewrites the JSONL files,
keeping the original label in `pred_original`, and recomputes the debate
aggregates (D0 and vote) from the corrected round labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import read_jsonl
from .runners import majority


def _rewrite(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def apply_corrections(results: Path, checked_csv: Path) -> pd.DataFrame:
    chk = pd.read_csv(checked_csv, dtype=str).fillna("")
    chk = chk[chk.manual_label.str.strip() != ""]
    fixes = {(r.task, r.condition, r.id, r.agent): r.manual_label.strip() for r in chk.itertuples()}
    log = []
    for f in sorted((Path(results) / "single_agent").glob("*.jsonl")):
        recs = read_jsonl(f)
        for r in recs:
            cond = f"{r['model']}|{r['method']}|p{r['paraphrase']}|s{r['seed']}"
            new = fixes.get((r["task"], cond, r["id"], ""))
            if new and new != r["pred"]:
                r.setdefault("pred_original", r["pred"]); log.append((f.name, cond, r["id"], r["pred"], new))
                r["pred"] = new
        _rewrite(f, recs)
    for f in sorted((Path(results) / "debate").glob("*.jsonl")):
        recs = read_jsonl(f)
        for r in recs:
            changed = False
            for k, rd in enumerate(r["rounds"]):
                cond = f"{r['composition']}|round{k}|s{r['seed']}"
                for i, a in enumerate(rd):
                    new = fixes.get((r["task"], cond, r["id"], str(i)))
                    if new and new != a["label"]:
                        a.setdefault("label_original", a["label"]); log.append((f.name, cond, r["id"], a["label"], new))
                        a["label"] = new; changed = True
            if changed:
                pool = [a["label"] for rd in r["rounds"] for a in rd]
                r["pred_D0"] = majority([a["label"] for a in r["rounds"][0]], pool)
                r["pred_vote"] = majority([a["label"] for a in r["rounds"][-1]], pool)
        _rewrite(f, recs)
    return pd.DataFrame(log, columns=["file", "condition", "id", "old", "new"])
