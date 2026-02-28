import random

from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v1/action_object_v2.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def validate(gpt_response, prompt=""):
  if len(gpt_response.strip()) < 1:
    return False
  return True

def clean_up(gpt_response, prompt=""):
  cleaned_response = gpt_response.strip()
  return cleaned_response

def fail_safe():
  fs = ("bed")
  return fs


def create_prompt_input(action_description,
                        persona,
                        temp_address,
                        test_input=None):
  prompt_input = []
  if "(" in action_description:
    action_description = action_description.split("(")[-1][:-1]

  prompt_input += [action_description]
  prompt_input += [persona
                   .s_mem.get_str_accessible_arena_game_objects(temp_address)]
  return prompt_input


def run_gpt_prompt_action_game_object(action_description,
                                      persona,
                                      maze,
                                      temp_address,
                                      test_input=None,
                                      verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(action_description,
                                     persona,
                                     temp_address,
                                     test_input)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)

  x = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
  if output not in x:
    output = random.choice(x)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
