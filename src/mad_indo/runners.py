"""Runners for single-agent prompting (S1-S3) and multi-agent debate (D0, debate, judge)."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from . import prompts
from .data import append_jsonl, done_ids
from .extract import extract_label


def _call(client, model, messages, labels, temperature, seed):
    out = client.chat(model, messages, temperature=temperature, seed=seed,
                      schema=prompts.json_schema(labels))
    ex = extract_label(out["content"], labels)
    return {**ex, "raw": out["content"], "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"], "latency_s": out["latency_s"],
            "load_s": out.get("load_s", 0.0), "error": out["error"]}


def majority(labels_list, tiebreak_pool=None):
    """Majority label; ties broken by frequency in tiebreak_pool, then by first occurrence."""
    c = Counter(labels_list)
    top = max(c.values())
    tied = [l for l in labels_list if c[l] == top]
    if len(set(tied)) == 1:
        return tied[0]
    if tiebreak_pool:
        c2 = Counter(l for l in tiebreak_pool if l in tied)
        best = max(c2.values())
        for l in tiebreak_pool:
            if l in tied and c2[l] == best:
                return l
    return tied[0]


def few_shot_examples(train_df: pd.DataFrame, per_class: int, seed: int):
    ex = pd.concat([g.sample(n=per_class, random_state=seed)
                    for _, g in train_df.groupby("label")])
    ex = ex.sample(frac=1.0, random_state=seed)  # shuffle so classes are interleaved
    return list(zip(ex["text"], ex["label"]))


def run_single_agent(client, model_key, model, task, labels, test_df, train_df,
                     setting, paraphrase, seed, cfg, out_path, workers=1):
    """Writes one JSONL record per sample; resumes if interrupted.
    workers > 1 sends that many requests to Ollama at once (needs OLLAMA_NUM_PARALLEL >= workers)."""
    done = done_ids(out_path)
    examples = None
    if setting == "S2_few_shot":
        examples = few_shot_examples(train_df, cfg["single_agent"]["few_shot_per_class"], seed)
    max_words = int(cfg["llm"]["max_rationale_tokens"] * 0.6)
    todo = [row for row in test_df.itertuples()
            if f"{task}|{model_key}|{setting}|p{paraphrase}|s{seed}|{row.id}" not in done]

    def work(row):
        msgs = prompts.single_agent_messages(setting, paraphrase, task, labels, row.text,
                                             examples=examples, max_words=max_words)
        res = _call(client, model, msgs, labels, cfg["single_agent"]["temperature"], seed)
        return {
            "run_key": f"{task}|{model_key}|{setting}|p{paraphrase}|s{seed}|{row.id}",
            "task": task, "method": setting, "model": model_key,
            "paraphrase": paraphrase, "seed": seed, "id": row.id, "gold": row.label,
            "pred": res["label"], "path": res["path"], "mentions_other": res["mentions_other"],
            "rationale": res["rationale"], "raw": res["raw"],
            "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
            "latency_s": res["latency_s"], "error": res["error"], "workers": workers,
        }

    _run_pool(work, todo, out_path, workers)


def _run_pool(work, items, out_path, workers):
    """Run work(item) for every item; results are written by this thread only (safe appends)."""
    if workers <= 1:
        for it in items:
            append_jsonl(out_path, work(it))
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, it) for it in items]
        for f in as_completed(futures):
            append_jsonl(out_path, f.result())


def _pmap(fn, items, workers):
    """Ordered parallel map."""
    if workers <= 1:
        return [fn(it) for it in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def run_debate(client, composition, agent_models, task, labels, test_df, seed, cfg,
               out_path, judge_model=None, workers=1, train_df=None):
    """composition: e.g. 'homo-qwen' or 'hetero'. agent_models: list of (key, ollama_tag).

    Rounds are processed phase by phase over chunks of samples: all round-0 calls of
    agent 1, then agent 2, ..., then each revision round, then the judge. The protocol
    (prompts, seeds, information each agent sees) is identical to sample-by-sample
    processing, but each model is loaded once per phase instead of once per call, which
    prevents GPU-memory thrashing when the agents use different models.
    Each record stores every round, so D0, debate + vote and debate + judge come from
    the same run. Records are written per chunk, so an interrupted run resumes.
    """
    done = done_ids(out_path)
    dcfg = cfg["debate"]
    max_words = int(cfg["llm"]["max_rationale_tokens"] * 0.6)
    temp = dcfg["temperature"]
    chunk = int(dcfg.get("chunk_size", 100))
    round0 = dcfg.get("round0_setting", "S1_zero_shot")
    examples = None
    if round0 == "S2_few_shot":
        assert train_df is not None, "round0_setting S2_few_shot needs train_df"
        # Same example set as single-agent S2 with this seed, so D0 and S2 are paired.
        examples = few_shot_examples(train_df, cfg["single_agent"]["few_shot_per_class"], seed)
    todo = [row for row in test_df.itertuples() if f"{task}|{composition}|s{seed}|{row.id}" not in done]
    n_agents = len(agent_models)

    for c0 in range(0, len(todo), chunk):
        rows = todo[c0:c0 + chunk]
        rounds = {row.id: [] for row in rows}
        # Round 0: independent answers, one model at a time (agent index added to the seed).
        r0 = {row.id: [None] * n_agents for row in rows}
        for i, (_, m) in enumerate(agent_models):
            res = _pmap(lambda row: _call(client, m, prompts.debate_round0_messages(
                task, labels, row.text, max_words, examples=examples), labels, temp, seed * 100 + i),
                rows, workers)
            for row, x in zip(rows, res):
                r0[row.id][i] = x
        for row in rows:
            rounds[row.id].append(r0[row.id])
        # Revision rounds for samples without unanimity.
        for r in range(1, dcfg["max_rounds"] + 1):
            active = [row for row in rows if len({a["label"] for a in rounds[row.id][-1]}) > 1]
            if not active:
                break
            cur = {row.id: [None] * n_agents for row in active}
            for i, (_, m) in enumerate(agent_models):
                def rev(row, i=i, m=m, r=r):
                    prev = rounds[row.id][-1]
                    others = [prev[j] for j in range(n_agents) if j != i]
                    msgs = prompts.debate_revision_messages(task, labels, row.text, prev[i], others, r, max_words)
                    return _call(client, m, msgs, labels, temp, seed * 100 + 10 * r + i)
                for row, x in zip(active, _pmap(rev, active, workers)):
                    cur[row.id][i] = x
            for row in active:
                rounds[row.id].append(cur[row.id])
        # Judge for every sample that was not unanimous at round 0.
        judged = {}
        if judge_model is not None:
            need = [row for row in rows if not (len(rounds[row.id]) == 1
                    and len({a["label"] for a in rounds[row.id][0]}) == 1)]
            def jud(row):
                tr = [[{"label": a["label"], "rationale": a["rationale"]} for a in rd] for rd in rounds[row.id]]
                return _call(client, judge_model, prompts.judge_messages(task, labels, row.text, tr, max_words),
                             labels, 0.0, seed)
            for row, x in zip(need, _pmap(jud, need, workers)):
                judged[row.id] = x
        for row in rows:
            append_jsonl(out_path, _debate_record(task, composition, seed, workers, round0, row,
                                                  agent_models, rounds[row.id], judge_model, judged.get(row.id)))


def _debate_record(task, composition, seed, workers, round0, row, agent_models, rounds, judge_model, jr):
    calls = [a for rd in rounds for a in rd]
    all_labels = [a["label"] for a in calls]
    d0 = majority([a["label"] for a in rounds[0]], tiebreak_pool=all_labels)
    vote = majority([a["label"] for a in rounds[-1]], tiebreak_pool=all_labels)
    judge = None
    if judge_model is not None:
        judge = jr["label"] if jr is not None else vote  # unanimous at round 0: nothing to judge
    return {
        "run_key": f"{task}|{composition}|s{seed}|{row.id}", "task": task, "composition": composition,
        "workers": workers, "round0": round0, "seed": seed, "phased": True,
        "id": row.id, "gold": row.label, "n_rounds": len(rounds) - 1,
        "pred_D0": d0, "pred_vote": vote, "pred_judge": judge,
        "rounds": [[{"agent": agent_models[i][0], "label": a["label"], "path": a["path"],
                     "mentions_other": a["mentions_other"], "rationale": a["rationale"]}
                    for i, a in enumerate(rd)] for rd in rounds],
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "latency_s": sum(c["latency_s"] for c in calls),
        "load_s": sum(c.get("load_s", 0.0) for c in calls),
        # D0 cost = round-0 calls only, so the cost frontier stays fair.
        "d0_prompt_tokens": sum(a["prompt_tokens"] for a in rounds[0]),
        "d0_completion_tokens": sum(a["completion_tokens"] for a in rounds[0]),
        "d0_latency_s": sum(a["latency_s"] for a in rounds[0]),
        "d0_load_s": sum(a.get("load_s", 0.0) for a in rounds[0]),
        # Judge call is logged separately: vote cost excludes it, judge cost adds it.
        "judge_prompt_tokens": jr["prompt_tokens"] if jr else 0,
        "judge_completion_tokens": jr["completion_tokens"] if jr else 0,
        "judge_latency_s": jr["latency_s"] if jr else 0.0,
        "judge_load_s": jr.get("load_s", 0.0) if jr else 0.0,
        "errors": [c["error"] for c in calls + ([jr] if jr else []) if c["error"]],
    }
