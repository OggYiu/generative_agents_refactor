"""
Deterministic tests for clean_up / validate / fail_safe functions.

No LLM calls — these test the parsing logic that converts raw LLM
responses into structured outputs.  Fast, reliable, runs in CI.

Usage:
    cd reverie/backend_server
    python -m pytest ../../eval/test_clean_up.py -v
"""
import sys
import os
import ast

# Add backend_server and eval dir to path
_eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_eval_dir, "..", "reverie", "backend_server"))
sys.path.insert(0, os.path.dirname(_eval_dir))

from manifest import EVAL_CASES

# ---------------------------------------------------------------------------
# Import clean_up / validate from each module referenced in the manifest
# ---------------------------------------------------------------------------
from persona.prompt_template.run_gpt_prompts.wake_up_hour import (
    clean_up as wake_up_clean,
    validate as wake_up_validate,
    fail_safe as wake_up_fail_safe,
)
from persona.prompt_template.run_gpt_prompts.generate_hourly_schedule import (
    clean_up as hourly_clean,
    validate as hourly_validate,
    fail_safe as hourly_fail_safe,
)
from persona.prompt_template.run_gpt_prompts.daily_plan import (
    clean_up as daily_clean,
    validate as daily_validate,
    fail_safe as daily_fail_safe,
)

# ---- Also import modules not yet in the manifest, to prove they're
#      importable and test their basic parsing. ----
from persona.prompt_template.run_gpt_prompts.action_arena import (
    clean_up as arena_clean, validate as arena_validate,
)
from persona.prompt_template.run_gpt_prompts.decide_to_react import (
    clean_up as react_clean, validate as react_validate,
)
from persona.prompt_template.run_gpt_prompts.event_triple import (
    clean_up as triple_clean, validate as triple_validate,
)
from persona.prompt_template.run_gpt_prompts.extract_keywords import (
    clean_up as kw_clean, validate as kw_validate,
)
from persona.prompt_template.run_gpt_prompts.pronunciatio import (
    clean_up as pron_clean, validate as pron_validate,
)


# ===========================================================================
#  Golden-file tests  (manifest-driven)
# ===========================================================================

def _read_eval_file(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class TestWakeUpHourCleanUp:
    """wake_up_hour: clean_up should parse '6 am' variants into int 6."""

    golden = int(_read_eval_file(EVAL_CASES["generate_wake_up_hour"]["golden_file"]))

    def test_exact_golden(self):
        assert wake_up_clean("6 am") == self.golden

    def test_with_whitespace(self):
        assert wake_up_clean("  6 am\n") == 6

    def test_uppercase(self):
        assert wake_up_clean("6 AM") == 6

    def test_seven_am(self):
        assert wake_up_clean("7 am") == 7

    def test_with_minutes(self):
        assert wake_up_clean("6:00 AM") == 6

    def test_with_minutes_lowercase(self):
        assert wake_up_clean("7:30 am") == 7

    def test_pm(self):
        assert wake_up_clean("10 pm") == 10

    def test_validate_good(self):
        assert wake_up_validate("6 am") is True

    def test_validate_bad(self):
        assert wake_up_validate("not a number am") is False

    def test_fail_safe_returns_int(self):
        assert isinstance(wake_up_fail_safe(), int)


class TestHourlyScheduleCleanUp:
    """generate_hourly_schedule: clean_up should strip and remove trailing period."""

    golden = _read_eval_file(EVAL_CASES["generate_hourly_schedule_01"]["golden_file"])

    def test_exact_golden(self):
        assert hourly_clean(self.golden) == self.golden

    def test_strip_whitespace(self):
        assert hourly_clean("  waking up and completing her morning routine  ") == self.golden

    def test_strip_trailing_period(self):
        assert hourly_clean("waking up and completing her morning routine.") == self.golden

    def test_validate_good(self):
        assert hourly_validate("some activity") is True

    def test_fail_safe_returns_str(self):
        assert isinstance(hourly_fail_safe(), str)


class TestDailyPlanCleanUp:
    """daily_plan: clean_up parses numbered list like '1) item, 2) item'."""

    golden = ast.literal_eval(
        _read_eval_file(EVAL_CASES["generate_first_daily_plan"]["golden_file"])
    )

    def test_validate_good(self):
        # The parser expects text before the first numbered item (from the
        # prompt template), and items ending with "," or "." before the next
        # number.  The last item is dropped because no trailing digit follows.
        sample = ("Isabella's plan:\n"
                  "1) wake up at 6:00 am,\n"
                  "2) eat breakfast at 7:00 am,\n"
                  "3) go to work at 8:00 am")
        assert daily_validate(sample) is True

    def test_fail_safe_returns_list(self):
        fs = daily_fail_safe()
        assert isinstance(fs, list)
        assert len(fs) > 0


# ===========================================================================
#  Non-manifest modules — basic parsing sanity checks
# ===========================================================================

class TestActionArena:
    def test_clean_up_splits_on_brace(self):
        assert arena_clean("kitchen}extra") == "kitchen"

    def test_validate_rejects_empty(self):
        assert arena_validate("") is False

    def test_validate_rejects_no_brace(self):
        assert arena_validate("kitchen") is False


class TestDecideToReact:
    def test_clean_up_option_3(self):
        assert react_clean("Answer: Option 3") == "3"

    def test_clean_up_option_1(self):
        assert react_clean("Answer: Option 1") == "1"

    def test_validate_good(self):
        assert react_validate("Answer: Option 2") is True

    def test_validate_bad(self):
        assert react_validate("Answer: Option 5") is False


class TestEventTriple:
    def test_clean_up_basic(self):
        result = triple_clean("is, sleeping) extra")
        assert result == ["is", "sleeping"]

    def test_validate_good(self):
        assert triple_validate("is, sleeping) extra") is True

    def test_validate_wrong_count(self):
        assert triple_validate("single) extra") is False


class TestExtractKeywords:
    def test_clean_up_basic(self):
        result = kw_clean("cafe, morning\nEmotive keywords:\nhappy, excited")
        assert "cafe" in result
        assert "morning" in result
        assert "happy" in result
        assert "excited" in result

    def test_clean_up_strips_period(self):
        result = kw_clean("word.\nEmotive keywords:\nfun.")
        assert "word" in result
        assert "fun" in result

    def test_validate_good(self):
        assert kw_validate("cafe\nEmotive keywords:\nhappy") is True


class TestPronunciatio:
    def test_clean_up_short(self):
        assert pron_clean("\U0001f60b") == "\U0001f60b"

    def test_clean_up_truncates(self):
        result = pron_clean("\U0001f60b\U0001f600\U0001f4a4\U0001f4a4")
        assert len(result) <= 3

    def test_validate_rejects_empty(self):
        assert pron_validate("") is False
