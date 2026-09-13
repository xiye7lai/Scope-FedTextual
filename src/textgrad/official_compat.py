"""Compatibility primitives adapted from the official FedTextGrad implementation.

Upstream: https://github.com/ubc-tea/FedTextGrad
Reference commit: 7ebebee1efbae08c9a6e3bbcf3095a00a1517607

The official repository vendors TextGrad and uses a forward/evaluation/backward/
TGD-step loop.  This module keeps that experiment protocol while using this
project's cached Responses backend instead of importing the full upstream stack.
"""

from __future__ import annotations


UPSTREAM_REPOSITORY = "https://github.com/ubc-tea/FedTextGrad"
UPSTREAM_COMMIT = "7ebebee1efbae08c9a6e3bbcf3095a00a1517607"

FORMATTING_INSTRUCTION = (
    "The last line of your response should be of the following format: "
    "'Answer: $VALUE' where VALUE is a numerical value."
)

GENERAL_FORMATTING_INSTRUCTION = (
    "The last line of your response must be exactly 'Answer: $VALUE', where VALUE "
    "is the requested number, option, word, or short phrase."
)

OFFICIAL_INITIAL_PROMPT = (
    "You will answer a reasoning question. Think step by step. "
    + FORMATTING_INSTRUCTION
)

GENERAL_INITIAL_PROMPT = (
    "You will answer a reasoning question. Think step by step. "
    + GENERAL_FORMATTING_INSTRUCTION
)

BACKWARD_SYSTEM_PROMPT = (
    "You are the feedback engine in a text optimization system. Give constructive, "
    "specific criticism that helps improve the system prompt on similar future examples. "
    "Describe strategies and changes, but do not write a replacement prompt. If the "
    "response and objective are already perfect, say that no change is needed."
)

TGD_SYSTEM_PROMPT = (
    "You improve a text variable using potentially noisy feedback. Return the complete "
    "replacement variable between <IMPROVED_VARIABLE> and </IMPROVED_VARIABLE> tags, "
    "with no text outside those tags."
)


def build_backward_prompt(system_prompt: str, examples: list[dict]) -> str:
    conversations = []
    for row in examples:
        conversations.append(
            "<CONVERSATION>\n"
            f"<LM_SYSTEM_PROMPT>{system_prompt}</LM_SYSTEM_PROMPT>\n"
            f"<LM_INPUT>{row['question']}</LM_INPUT>\n"
            f"<LM_OUTPUT>{row['prediction']}</LM_OUTPUT>\n"
            f"<GROUND_TRUTH>{row['answer']}</GROUND_TRUTH>\n"
            f"<OBJECTIVE_SCORE>{int(row['correct'])}</OBJECTIVE_SCORE>\n"
            "</CONVERSATION>"
        )
    return (
        BACKWARD_SYSTEM_PROMPT
        + "\n\nThe variable is the system prompt below. Analyze the conversations and give "
        "feedback that improves answer accuracy.\n"
        f"<VARIABLE>{system_prompt}</VARIABLE>\n\n"
        + "\n\n".join(conversations)
    )


def build_tgd_prompt(system_prompt: str, feedback: str) -> str:
    return (
        TGD_SYSTEM_PROMPT
        + "\n\n<ROLE>structured system prompt for a reasoning QA model</ROLE>\n"
        f"<VARIABLE>{system_prompt}</VARIABLE>\n"
        f"<FEEDBACK>{feedback}</FEEDBACK>\n"
        "Improve the variable using the feedback while retaining its required answer format.\n"
        "<IMPROVED_VARIABLE>{complete improved prompt}</IMPROVED_VARIABLE>"
    )


def summarization_prompt(
    prompts: list[str],
    uid: bool = False,
    formatting_instruction: str = FORMATTING_INSTRUCTION,
) -> str:
    principle = (
        " Apply Uniform Information Density principles so important details are distributed "
        "evenly and no client prompt is over-compressed."
        if uid else ""
    )
    merged = "\n\n".join(f"<CLIENT_PROMPT_{i + 1}>{prompt}</CLIENT_PROMPT_{i + 1}>" for i, prompt in enumerate(prompts))
    return (
        "Merge the client prompts into one cohesive system prompt while preserving all "
        "useful original information."
        + principle
        + f" Preserve this required answer-format instruction as the final sentence: {formatting_instruction} "
        "Return only the merged prompt.\n\n"
        + merged
    )


def extract_improved_variable(text: str, fallback: str) -> str:
    start, end = "<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"
    if start in text and end in text:
        return text.split(start, 1)[1].split(end, 1)[0].strip()
    # Gateways occasionally omit tags. Keeping a non-empty replacement is more
    # robust, while the raw response remains cached for audit.
    cleaned = text.strip()
    return cleaned or fallback
