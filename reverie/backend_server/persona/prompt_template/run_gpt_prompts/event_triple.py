from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 30,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
PROMPT_TEMPLATE = "persona/prompt_template/v2/generate_event_triple_v1.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def clean_up(gpt_response, prompt=""):
  cr = gpt_response.strip()
  cr = [i.strip() for i in cr.split(")")[0].split(",")]
  return cr

def validate(gpt_response, prompt=""):
  try:
    gpt_response = clean_up(gpt_response, prompt="")
    if len(gpt_response) != 2:
      return False
  except: return False
  return True

def fail_safe(persona):
  fs = (persona.name, "is", "idle")
  return fs


def create_prompt_input(action_description, persona):
  if "(" in action_description:
    action_description = action_description.split("(")[-1].split(")")[0]
  prompt_input = [persona.name,
                  action_description,
                  persona.name]
  return prompt_input


def run_gpt_prompt_event_triple(action_description, persona, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(action_description, persona)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe(persona) ########
  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)
  output = (persona.name, output[0], output[1])

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
