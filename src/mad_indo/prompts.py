"""Prompt templates. Instructions are in English; input texts stay in Indonesian.

Each single-agent setting has three paraphrases (index 0, 1, 2) to measure
sensitivity to wording. Every prompt asks for a JSON object with separate
`label` and `rationale` fields, so the label never has to be found in free text.
"""
from __future__ import annotations

TASK_DESC = {
    "smsa": "the sentiment expressed in an Indonesian text",
    "emot": "the main emotion expressed in an Indonesian tweet",
}

OUTPUT_RULE = (
    'Reply with JSON only: {{"label": <one of {labels}>, "rationale": <at most {max_words} words>}}.'
)

S1_TEMPLATES = [
    "Classify {task}. Valid labels: {labels}.\n{rule}\n\nText: {text}",
    "You are an annotator. Decide {task}, choosing exactly one label from {labels}.\n{rule}\n\nText: {text}",
    "Read the text below and assign one label from {labels} that best describes {task}.\n{rule}\n\nText: {text}",
]

S2_TEMPLATES = [
    "Classify {task}. Valid labels: {labels}.\n{rule}\n\nExamples:\n{examples}\n\nText: {text}",
    "You are an annotator. Decide {task}, choosing exactly one label from {labels}. "
    "Labelled examples are given first.\n{rule}\n\n{examples}\n\nText: {text}",
    "Study the labelled examples, then assign one label from {labels} that best describes {task}.\n"
    "{rule}\n\n{examples}\n\nText: {text}",
]

S3_TEMPLATES = [
    "Classify {task}. Valid labels: {labels}. Think step by step in the rationale before choosing.\n"
    "{rule}\n\nText: {text}",
    "You are an annotator. Reason briefly about the wording and tone, then decide {task} "
    "using exactly one label from {labels}.\n{rule}\n\nText: {text}",
    "First explain your reasoning in a few steps, then give one label from {labels} for {task}.\n"
    "{rule}\n\nText: {text}",
]

TEMPLATES = {"S1_zero_shot": S1_TEMPLATES, "S2_few_shot": S2_TEMPLATES, "S3_cot": S3_TEMPLATES}


def _rule(labels, max_words):
    return OUTPUT_RULE.format(labels=" | ".join(labels), max_words=max_words)


def format_examples(examples) -> str:
    """examples: iterable of (text, label)."""
    return "\n".join(
        f'Text: {t}\nAnswer: {{"label": "{l}", "rationale": "..."}}' for t, l in examples
    )


def single_agent_messages(setting, paraphrase, task, labels, text, examples=None, max_words=60):
    tpl = TEMPLATES[setting][paraphrase]
    content = tpl.format(
        task=TASK_DESC[task],
        labels=", ".join(labels),
        rule=_rule(labels, max_words),
        text=text,
        examples=format_examples(examples or []),
    )
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------- debate
def debate_round0_messages(task, labels, text, max_words=60, examples=None):
    """Round 0 uses paraphrase 0 of the single-agent prompt so that D0 is directly comparable:
    with examples -> S2 few-shot (same example set as S2 for that seed); without -> S1 zero-shot."""
    if examples:
        return single_agent_messages("S2_few_shot", 0, task, labels, text, examples=examples, max_words=max_words)
    return single_agent_messages("S1_zero_shot", 0, task, labels, text, max_words=max_words)


def debate_revision_messages(task, labels, text, own, others, round_idx, max_words=60):
    """own/others: dicts with 'label' and 'rationale'. Others are anonymised as Agent A, B, ..."""
    peer_block = "\n".join(
        f"Agent {chr(65 + i)}: label = {o['label']}; rationale = {o['rationale']}"
        for i, o in enumerate(others)
    )
    content = (
        f"You are taking part in a discussion to decide {TASK_DESC[task]}. "
        f"Valid labels: {', '.join(labels)}.\n\n"
        f"Text: {text}\n\n"
        f"Your previous answer: label = {own['label']}; rationale = {own['rationale']}\n\n"
        f"Answers from the other agents in the previous round:\n{peer_block}\n\n"
        f"This is revision round {round_idx}. Examine the other arguments critically. "
        "Keep your label if your reasoning still holds, or change it if another argument is better.\n"
        + _rule(labels, max_words)
    )
    return [{"role": "user", "content": content}]


def judge_messages(task, labels, text, transcript, max_words=60):
    """transcript: list of rounds; each round is a list of {'label','rationale'} (anonymised)."""
    lines = []
    for r, answers in enumerate(transcript):
        for i, a in enumerate(answers):
            lines.append(f"Round {r}, Agent {chr(65 + i)}: label = {a['label']}; rationale = {a['rationale']}")
    content = (
        f"You are the judge of a discussion about {TASK_DESC[task]}. "
        f"Valid labels: {', '.join(labels)}.\n\nText: {text}\n\nDiscussion:\n"
        + "\n".join(lines)
        + "\n\nWeigh the arguments, not the number of agents, and give the final label.\n"
        + _rule(labels, max_words)
    )
    return [{"role": "user", "content": content}]


def json_schema(labels):
    """Ollama structured-output schema: the label field is restricted to valid labels."""
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "rationale": {"type": "string"},
        },
        "required": ["label", "rationale"],
    }
