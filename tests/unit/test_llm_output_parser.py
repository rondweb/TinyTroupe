from tinytroupe.llm.output_parser import fallback_action, parse_action_json


def test_parse_action_json_normalizes_action_defaults():
    payload = parse_action_json(
        '{"action": {"type": "TALK"}, "cognitive_state": {"goals": "g", "attention": "a", "emotions": "e"}}'
    )

    assert payload["action"]["type"] == "TALK"
    assert payload["action"]["content"] == ""
    assert payload["action"]["target"] is None


def test_parse_action_json_adds_default_cognitive_state():
    payload = parse_action_json('{"action": {"type": "DONE"}}')

    assert payload["cognitive_state"]["goals"] == "Continue current interaction"
    assert payload["cognitive_state"]["attention"] == "Current conversation"
    assert payload["cognitive_state"]["emotions"] == "Neutral"


def test_parse_action_json_raises_when_action_keys_absent():
    try:
        parse_action_json('{"foo": "bar"}')
        assert False, "Expected ValueError when action keys are missing"
    except ValueError as exc:
        assert "Missing both 'action' and 'actions'" in str(exc)


def test_fallback_action_detects_talk():
    payload = fallback_action("I should TALK to the user now.")

    assert payload["action"]["type"] == "TALK"
    assert payload["action"]["target"] is None


def test_fallback_action_defaults_to_done():
    payload = fallback_action("unstructured text")

    assert payload["action"]["type"] == "DONE"
