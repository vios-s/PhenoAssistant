"""
utils/verifier.py
=================
Contribution 2 — Self-Verification Agent

After the tool selector picks a tool, a second lightweight LLM call
verifies the selection is logically consistent with the user query.

This catches cases where:
- Retrieval returns the right tool at rank #1
- But the manager picks the wrong one from the shortlist

Architecture position:
  Manager picks tool from top-k
      ↓
  Verifier checks: does this tool match the task?   ← this file
      ↓
  ACCEPT → execute
  REJECT → fall back to rank #1 from retrieval
"""

import os


def verify_tool_selection(
    user_query: str,
    selected_tool: str,
    tool_description: str,
    client=None,
    model_name: str = None,
) -> dict:
    """
    Verify that the selected tool is appropriate for the user query.

    Returns
    -------
    dict with keys:
        accepted    : bool   — True if selection is valid
        confidence  : int    — 1-10
        reason      : str    — short explanation
    """
    from openai import AzureOpenAI

    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_API_URL"],
            api_version=os.environ["AZURE_API_VERSION"],
        )
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]

    prompt = (
        f"User task: '{user_query}'\n\n"
        f"Selected tool: {selected_tool}\n"
        f"Tool description: {tool_description}\n\n"
        "Does this tool correctly match the user's task?\n"
        "Reply in this exact format:\n"
        "verdict:YES or NO\n"
        "confidence:1-10\n"
        "reason:one short sentence\n"
        "Nothing else."
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quality-checking agent for a plant phenotyping AI system. "
                    "Your job is to verify that the selected tool matches the user's intent. "
                    "Be strict — reject if there is any mismatch."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=60,
    )

    raw = resp.choices[0].message.content.strip()
    result = {"accepted": False, "confidence": 5, "reason": raw, "raw": raw}

    for line in raw.lower().splitlines():
        if line.startswith("verdict:"):
            result["accepted"] = "yes" in line
        elif line.startswith("confidence:"):
            try:
                result["confidence"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("reason:"):
            result["reason"] = line.split(":", 1)[1].strip()

    status = "ACCEPTED" if result["accepted"] else "REJECTED"
    print(f"[Verifier] {status} '{selected_tool}' "
          f"(confidence: {result['confidence']}/10) — {result['reason']}")
    return result


def verify_model_selection(
    user_query: str,
    selected_model_id: str,
    crop: str,
    task: str,
    modality: str,
    client=None,
    model_name: str = None,
) -> dict:
    """
    Verify the model selector's (crop, task, modality) triple
    is consistent with the user query before looking up the model.
    """
    from openai import AzureOpenAI

    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_API_URL"],
            api_version=os.environ["AZURE_API_VERSION"],
        )
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]

    prompt = (
        f"User task: '{user_query}'\n\n"
        f"Decision tree answers:\n"
        f"  Crop:     {crop}\n"
        f"  Task:     {task}\n"
        f"  Modality: {modality}\n"
        f"  Selected model: {selected_model_id}\n\n"
        "Are these answers consistent with the user task?\n"
        "Reply in this exact format:\n"
        "verdict:YES or NO\n"
        "confidence:1-10\n"
        "reason:one short sentence\n"
        "Nothing else."
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a verification agent for a plant phenotyping AI system. "
                    "Check that crop, task, and modality are all consistent with the user query."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=60,
    )

    raw = resp.choices[0].message.content.strip()
    result = {"accepted": False, "confidence": 5, "reason": raw}

    for line in raw.lower().splitlines():
        if line.startswith("verdict:"):
            result["accepted"] = "yes" in line
        elif line.startswith("confidence:"):
            try:
                result["confidence"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("reason:"):
            result["reason"] = line.split(":", 1)[1].strip()

    status = "ACCEPTED" if result["accepted"] else "REJECTED"
    print(f"[Verifier] {status} ({crop}, {task}, {modality}) → {selected_model_id} "
          f"(confidence: {result['confidence']}/10) — {result['reason']}")
    return result
