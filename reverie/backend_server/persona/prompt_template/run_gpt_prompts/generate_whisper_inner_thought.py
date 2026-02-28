from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 50,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def create_prompt_input(persona, whisper, test_input=None):
  prompt_input = [persona.scratch.name, whisper]
  return prompt_input


def clean_up(gpt_response, prompt=""):
  return gpt_response.split('"')[0].strip()

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False

def fail_safe():
  return "..."


def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(persona, whisper)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
