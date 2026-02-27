"""
Quick eval: send the hourly schedule prompt to the current LLM provider
using completion_request (same as the real code path) and compare against
the golden output.
"""
import sys
import os

# Add backend_server to path so we can import the LLM provider
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))

from persona.prompt_template.llm_provider import completion_request
from persona.prompt_template.gpt_structure import generate_prompt
from utils import llm_provider, vscode_base_url, vscode_model

# Read golden output
golden_path = os.path.join(os.path.dirname(__file__), "generate_hourly_schedule_01_golden.txt")
with open(golden_path, "r", encoding="utf-8") as f:
    golden = f.read().strip()

# Build prompt from the v3 template using the same inputs as the prompt file
# The prompt file was generated with these inputs, so we reconstruct them.
prompt_file = os.path.join(os.path.dirname(__file__), "generate_hourly_schedule_01_prompt.txt")
with open(prompt_file, "r", encoding="utf-8") as f:
    raw_prompt = f.read()

# Use v3 template: parse the raw prompt to extract the input variables
# INPUT 0: schedule format (lines 1-25 of the raw prompt, after "Hourly schedule format:\n")
# INPUT 1: persona info block (lines 27-33)
# INPUT 2: prior schedule (lines 37-42)
# INPUT 3: intermission (line 45)
# INPUT 4: intermission2 (empty)
# INPUT 5: prompt ending (line 46, incomplete)

# Rather than re-parsing, let's just build the v3 prompt by prepending the
# instruction to the existing raw prompt content.

template_path = os.path.join(
    os.path.dirname(__file__), "..",
    "reverie", "backend_server",
    "persona", "prompt_template", "v2", "generate_hourly_schedule_v3.txt"
)

# Read the v3 template to get just the instruction header
with open(template_path, "r", encoding="utf-8") as f:
    template_content = f.read()

# Extract the instruction part (everything between <commentblockmarker> and the template body)
instruction = """Task: Complete the last line of an hourly activity schedule for a character. Output ONLY the brief activity description that finishes the last line. Do not continue the schedule beyond that line. Do not include timestamps, IDs, or the character's name.

Example output: waking up and getting ready for the day

"""

prompt = instruction + raw_prompt

gpt_param = {
    "engine": "text-davinci-003",
    "max_tokens": 50,
    "temperature": 0.5,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["\n"]
}

print(f"LLM Provider: {llm_provider}")
if llm_provider == "vscode":
    print(f"  Base URL: {vscode_base_url}")
    print(f"  Model: {vscode_model}")
print(f"Golden output: \"{golden}\"")
print("---")
print("Sending prompt to LLM (v3 template with instructions)...")

try:
    response = completion_request(prompt, gpt_param)
    # Clean up: strip, remove trailing period
    cleaned = response.strip()
    if cleaned and cleaned[-1] == ".":
        cleaned = cleaned[:-1]
    print(f"LLM response (raw):     \"{response}\"")
    print(f"LLM response (cleaned): \"{cleaned}\"")
    print("---")

    # Check similarity
    response_lower = cleaned.lower().strip()
    golden_lower = golden.lower().strip()

    if golden_lower == response_lower:
        print("RESULT: EXACT MATCH")
    elif golden_lower in response_lower:
        print("RESULT: PASS - Golden output found in response")
    elif response_lower in golden_lower:
        print("RESULT: PASS - Response is a subset of golden output")
    else:
        golden_words = set(golden_lower.split())
        response_words = set(response_lower.split())
        overlap = golden_words & response_words
        overlap_ratio = len(overlap) / len(golden_words) if golden_words else 0
        print(f"Keyword overlap: {len(overlap)}/{len(golden_words)} = {overlap_ratio:.0%}")
        print(f"  Matching words: {overlap}")
        print(f"  Missing words: {golden_words - response_words}")
        if overlap_ratio >= 0.5:
            print("RESULT: CLOSE - Significant keyword overlap with similar meaning")
        else:
            print("RESULT: DIFFERENT - Low keyword overlap")

    # Also check it's a single short phrase (not a full schedule)
    lines = [l for l in response.strip().split("\n") if l.strip()]
    if len(lines) == 1:
        print("FORMAT: OK - Single line output")
    else:
        print(f"FORMAT: WARN - Got {len(lines)} lines (expected 1)")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
