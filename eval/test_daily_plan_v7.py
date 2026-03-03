"""
Quick interactive test for daily_planning_v7.txt.

Run from reverie/backend_server:
    python ../../eval/test_daily_plan_v7.py

Shows the raw LLM response and how clean_up parses it, so you can iterate
on the prompt template without running the full simulation.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from persona.prompt_template.gpt_structure import generate_prompt, GPT_request
from persona.prompt_template.run_gpt_prompts.daily_plan import (
    clean_up, validate, GPT_PARAM, PROMPT_TEMPLATE,
)

# Isabella Rodriguez inputs — same persona as the existing eval fixtures
PROMPT_INPUTS = [
    # INPUT 0: commonset (get_str_iss output)
    ("Name: Isabella Rodriguez\n"
     "Age: 34\n"
     "Innate traits: friendly, outgoing, hospitable\n"
     "Learned traits: Isabella Rodriguez is a cafe owner of Hobbs Cafe who loves "
     "to make people feel welcome. She is always looking for ways to make the cafe "
     "a place where people can come to relax and enjoy themselves.\n"
     "Currently: Isabella Rodriguez is planning on having a Valentine's Day party "
     "at Hobbs Cafe with her customers on February 14th, 2023 at 5pm. She is "
     "gathering party material, and is telling everyone to join the party at Hobbs "
     "Cafe on February 14th, 2023, from 5pm to 7pm.\n"
     "Lifestyle: Isabella Rodriguez goes to bed around 11pm, awakes up around 6am.\n"
     "Daily plan requirement: Isabella Rodriguez opens Hobbs Cafe at 8am everyday, "
     "and works at the counter until 8pm, at which point she closes the cafe."),
    # INPUT 1: lifestyle (get_str_lifestyle output)
    "Isabella Rodriguez goes to bed around 11pm, awakes up around 6am.",
    # INPUT 2: current date
    "Monday February 13",
    # INPUT 3: first name
    "Isabella",
    # INPUT 4: wake_up_hour formatted
    "6:00 am",
]

print(f"Using template: {PROMPT_TEMPLATE}")
print()

prompt = generate_prompt(PROMPT_INPUTS, PROMPT_TEMPLATE)

print("=" * 60)
print("PROMPT SENT TO LLM")
print("=" * 60)
print(prompt)
print()

TEST_PARAM = {**GPT_PARAM, "temperature": 0.5}
raw = GPT_request(prompt, TEST_PARAM)

print("=" * 60)
print("RAW LLM RESPONSE")
print("=" * 60)
print(repr(raw))
print()
print(raw)
print()

is_valid = validate(raw)
print(f"validate() => {is_valid}")
print()

if is_valid:
    parsed = clean_up(raw)
    full_plan = ["wake up and complete the morning routine at 6:00 am"] + parsed

    print("=" * 60)
    print("PARSED DAILY PLAN")
    print("=" * 60)
    for idx, item in enumerate(full_plan, 1):
        print(f"  {idx}) {item}")
else:
    print("PARSE FAILED — response did not validate.")
    print("(clean_up returned empty list or raised an exception)")
