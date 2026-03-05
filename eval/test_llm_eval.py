"""
LLM evaluation tests — sends prompts to the LLM via the /v1/completions
endpoint (legacy text completion), runs the response through the module's
clean_up function, and compares against the golden output.

Uses the vscode proxy's /v1/completions endpoint for raw text completion
(just the continuation), matching what the original text-davinci models
produced.  Falls back to Ollama's /api/generate if configured.

These are SLOW and NON-DETERMINISTIC.  Run manually, not in CI.

Usage:
    cd reverie/backend_server
    python -m pytest ../../eval/test_llm_eval.py -v -s

Or run a single case:
    python -m pytest ../../eval/test_llm_eval.py -v -s -k "wake_up_hour"

Configure via env vars (falls back to utils.py defaults):
    EVAL_BASE_URL=http://127.0.0.1:3030/v1  EVAL_MODEL=gpt-4o
"""
import re
import sys
import os
import ast
import importlib

import pytest
from openai import OpenAI

# Add backend_server and eval dir to path
_eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_eval_dir, "..", "reverie", "backend_server"))
sys.path.insert(0, os.path.dirname(_eval_dir))

from manifest import EVAL_CASES

# ---------------------------------------------------------------------------
# Raw text completion via /v1/completions — returns just the continuation
# tokens, matching what the original text-davinci models produced.
# ---------------------------------------------------------------------------
from utils import (
    vscode_base_url as _default_url,
    vscode_model as _default_model,
)

_BASE_URL = os.environ.get("EVAL_BASE_URL", _default_url)
_MODEL = os.environ.get("EVAL_MODEL", _default_model)

_client = OpenAI(base_url=_BASE_URL, api_key="no-key-needed")


def _completion_request(prompt, gpt_param):
    """Send prompt via /v1/completions and return raw continuation text."""
    resp = _client.completions.create(
        model=_MODEL,
        prompt=prompt,
        max_tokens=gpt_param.get("max_tokens", 256),
        temperature=gpt_param.get("temperature", 0),
        top_p=gpt_param.get("top_p", 1),
        stop=gpt_param.get("stop", None),
    )
    return resp.choices[0].text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EVAL_DIR = os.path.dirname(__file__)


def _read_file(filename):
    path = os.path.join(EVAL_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_module(module_path):
    """Dynamically import a module and return it."""
    return importlib.import_module(module_path)


def _parse_golden(raw_golden, golden_type):
    """Convert the raw golden string to the expected Python type."""
    if golden_type == "int":
        return int(raw_golden)
    elif golden_type == "str":
        return raw_golden
    elif golden_type == "list":
        return ast.literal_eval(raw_golden)
    else:
        return raw_golden


def _strip_markdown(text):
    """Remove common markdown formatting that chat-trained models add."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic*
    text = re.sub(r'`(.+?)`', r'\1', text)         # `code`
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # # headers
    return text.strip()


def _keyword_overlap(expected_str, actual_str):
    """Return the fraction of expected keywords found in actual."""
    expected_words = set(expected_str.lower().split())
    actual_words = set(actual_str.lower().split())
    if not expected_words:
        return 1.0
    return len(expected_words & actual_words) / len(expected_words)


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(EVAL_CASES.keys()), ids=list(EVAL_CASES.keys()))
def eval_case(request):
    """Yields (case_name, case_config) for each manifest entry."""
    name = request.param
    return name, EVAL_CASES[name]


def test_llm_prompt_against_golden(eval_case):
    """
    For each eval case:
      1. Read the prompt file
      2. Send it to the LLM
      3. Check if validate() accepts the response
      4. Run through clean_up and compare against golden
    """
    case_name, config = eval_case
    mod = _load_module(config["module"])

    # Read files
    raw_prompt = _read_file(config["prompt_file"])
    raw_golden = _read_file(config["golden_file"])
    golden = _parse_golden(raw_golden, config["golden_type"])

    # Send to LLM
    gpt_param = config["gpt_param"]
    print(f"\n--- {case_name} ---")
    print(f"Golden: {golden!r}")

    raw_response = _completion_request(raw_prompt, gpt_param)
    response = _strip_markdown(raw_response)
    print(f"LLM raw response: {raw_response!r}")
    print(f"After strip_markdown: {response!r}")

    # --- Layer 1: validate() ---
    # This is the same check the real code does before accepting a response.
    is_valid = mod.validate(response, prompt=raw_prompt)
    print(f"validate() passed: {is_valid}")

    if not is_valid:
        # validate() rejected — this means the real code path would retry
        # or fall back to fail_safe.  Report it clearly.
        print(f"RESULT: VALIDATE_FAILED — clean_up can't parse this response")
        print(f"  This usually means your LLM provider returns a different")
        print(f"  format than expected (e.g. chat model echoing context vs")
        print(f"  completion model returning just the answer).")
        print(f"  Golden value '{raw_golden}' present in raw response: "
              f"{raw_golden in response}")
        pytest.fail(
            f"validate() rejected the LLM response for {case_name}. "
            f"Raw response: {response!r}. "
            f"The clean_up parser expects a different format. "
            f"Golden value present in raw: {raw_golden in response}"
        )

    # --- Layer 2: clean_up + comparison ---
    # Pass the prompt so modules like task_decomp can extract context from it
    cleaned = mod.clean_up(response, prompt=raw_prompt)
    print(f"After clean_up: {cleaned!r}")

    if config["golden_type"] == "int":
        assert cleaned == golden, (
            f"Expected {golden}, got {cleaned}"
        )
        print("RESULT: EXACT MATCH")

    elif config["golden_type"] == "str":
        golden_lower = str(golden).lower().strip()
        cleaned_lower = str(cleaned).lower().strip()

        if golden_lower == cleaned_lower:
            print("RESULT: EXACT MATCH")
        elif golden_lower in cleaned_lower or cleaned_lower in golden_lower:
            print("RESULT: PASS — substring match")
        else:
            overlap = _keyword_overlap(golden_lower, cleaned_lower)
            print(f"RESULT: keyword overlap = {overlap:.0%}")
            assert overlap >= 0.5, (
                f"Low keyword overlap ({overlap:.0%}). "
                f"Expected: {golden!r}, Got: {cleaned!r}"
            )

    elif config["golden_type"] == "raw":
        # Compare the raw LLM response directly against golden (no clean_up).
        # Used when the meaningful ground truth is the model's output format
        # itself (e.g. task_decomp), not the parsed structure.
        raw_lower = response.lower().strip()
        golden_lower = str(golden).lower().strip()
        if golden_lower == raw_lower:
            print("RESULT: EXACT MATCH")
        elif golden_lower in raw_lower or raw_lower in golden_lower:
            print("RESULT: PASS — substring match")
        else:
            overlap = _keyword_overlap(golden_lower, raw_lower)
            print(f"RESULT: keyword overlap = {overlap:.0%}")
            assert overlap >= 0.5, (
                f"Low keyword overlap ({overlap:.0%}). "
                f"Expected: {golden!r}, Got: {response!r}"
            )

    elif config["golden_type"] == "list":
        assert isinstance(cleaned, list), f"Expected list, got {type(cleaned)}"
        print(f"Golden has {len(golden)} items, result has {len(cleaned)} items")

    else:
        assert cleaned is not None and cleaned != "", (
            f"clean_up returned empty for {case_name}"
        )
