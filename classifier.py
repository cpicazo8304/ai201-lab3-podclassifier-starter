import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    def format_example(example: dict) -> str:
        title = example.get("title", "").strip()
        description_text = example.get("description", "").strip()
        label = example.get("label", "").strip()

        example_lines = []
        if title:
            example_lines.append(f"Title: {title}")
        example_lines.append(f"Description: {description_text}")
        example_lines.append(f"Label: {label}")
        return "\n".join(example_lines)

    task_instruction = (
        "You are classifying podcast episodes by their format. Classify the episode "
        "into exactly one of these four labels:\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — no guests, "
        "no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often debating or "
        "discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, archival "
        "audio, reporting — with a clear narrative arc\n\n"
        "Return only the label and your reasoning. Do not explain the taxonomy. "
        "Use only one of these labels: interview, solo, panel, narrative."
    )

    prompt_parts = [task_instruction]

    if labeled_examples:
        prompt_parts.append("Here are a few labeled examples:")
        prompt_parts.append("\n\n---\n\n".join(format_example(example) for example in labeled_examples))
        prompt_parts.append("\n\nUse these examples to infer the format pattern.")
    else:
        prompt_parts.append(
            "No labeled training examples are available. Classify based on the label "
            "definitions above."
        )

    prompt_parts.append(
        "\n\nNow classify the following episode description. The episode "
        "description may be short. Use the text that is available and choose the best label."
    )
    prompt_parts.append(f"\n\nDescription: {description.strip()}")
    prompt_parts.append(
        "\n\nReturn your answer in the exact format below:\n"
        "Label: <one label>\n"
        "Reasoning: <brief explanation>"
    )
    prompt_parts.append(
        "\n\nDo not add any extra text beyond the requested format."
    )

    return "\n\n".join(prompt_parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        # Extract text from the LLM response
        raw = response.choices[0].message.content

        # Parse for lines like 'Label: <label>' and 'Reasoning: <text>'
        label = None
        reasoning = ""
        for line in (raw or "").splitlines():
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("label:") and label is None:
                label = line.split(":", 1)[1].strip().lower()
            elif lower.startswith("reasoning:") and not reasoning:
                reasoning = line.split(":", 1)[1].strip()
        # If label wasn't found in header lines, try a looser parse (first token)
        if not label:
            # sometimes the model may return just the label on the first line
            first_line = (raw or "").splitlines()[0].strip() if (raw or "") else ""
            if first_line:
                # take first word and normalize
                candidate = first_line.split()[0].strip().lower()
                if candidate:
                    label = candidate

        if label not in VALID_LABELS:
            validated_label = "unknown"
        else:
            validated_label = label

        if not reasoning:
            # attempt to capture remaining text as reasoning
            parts = (raw or "").split("Reasoning:", 1)
            if len(parts) > 1:
                reasoning = parts[1].strip()
            else:
                # fallback to entire response
                reasoning = (raw or "").strip()
                
        return {"label": validated_label, "reasoning": reasoning}
    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error calling LLM: {e}"}
