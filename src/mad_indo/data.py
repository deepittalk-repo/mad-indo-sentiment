"""Data loading, configuration and result I/O helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "experiment.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(task: str, split: str, limit: int | None = None) -> pd.DataFrame:
    """Load a processed split with columns id, text, label."""
    df = pd.read_csv(DATA_DIR / "processed" / f"{task}_{split}.csv")
    if limit:
        # Stratified, deterministic subset so that pilot runs keep every class.
        parts = [g.sample(n=max(1, round(limit * len(g) / len(df))), random_state=0)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).sort_values("id").reset_index(drop=True)
    return df


def verify_checksums() -> pd.DataFrame:
    """Compare every data file against data/SHA256SUMS."""
    rows = []
    for line in (DATA_DIR / "SHA256SUMS").read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        actual = hashlib.sha256((DATA_DIR / rel).read_bytes()).hexdigest()
        rows.append({"file": rel, "ok": actual == expected})
    return pd.DataFrame(rows)


def results_dir(base: str | os.PathLike | None = None) -> Path:
    """Results go to Google Drive on Colab (if mounted) so runs survive disconnects."""
    if base is None:
        drive = Path("/content/drive/MyDrive/mad-indo-results")
        base = drive if Path("/content/drive/MyDrive").exists() else REPO_ROOT / "results"
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_ids(path: Path, key: str = "run_key") -> set:
    """Keys already written, so an interrupted run can resume."""
    return {r[key] for r in read_jsonl(path)}
