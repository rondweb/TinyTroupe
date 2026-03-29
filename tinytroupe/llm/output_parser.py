from typing import Any, Dict

from tinytroupe import utils


def parse_action_json(text: str) -> Dict[str, Any]:
    """
    Parse and normalize the TinyTroupe action JSON payload from model text.
    """
    data = utils.extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")

    if "action" not in data and "actions" not in data:
        raise ValueError("Missing both 'action' and 'actions' in model output")

    if "cognitive_state" not in data:
        data["cognitive_state"] = {
            "goals": "Continue current interaction",
            "attention": "Current conversation",
            "emotions": "Neutral",
        }

    if "action" in data:
        action = data["action"]
        if isinstance(action, str):
            action = {"type": action, "content": "", "target": None}
        if isinstance(action, dict):
            action.setdefault("content", "")
            action.setdefault("target", None)
            if "type" not in action:
                raise ValueError("Missing action.type")
        data["action"] = action

    return data


def fallback_action(text: str) -> Dict[str, Any]:
    upper_text = (text or "").upper()
    if "DONE" in upper_text:
        return {
            "action": {"type": "DONE", "content": "Completed action", "target": None},
            "cognitive_state": {
                "goals": "Task completed",
                "attention": "Finished",
                "emotions": "Satisfied",
            },
        }
    if "TALK" in upper_text:
        return {
            "action": {"type": "TALK", "content": text[:300], "target": None},
            "cognitive_state": {
                "goals": "Communicate",
                "attention": "Conversation",
                "emotions": "Engaged",
            },
        }

    return {
        "action": {"type": "DONE", "content": "Default completion", "target": None},
        "cognitive_state": {
            "goals": "Finish",
            "attention": "Done",
            "emotions": "Neutral",
        },
    }
