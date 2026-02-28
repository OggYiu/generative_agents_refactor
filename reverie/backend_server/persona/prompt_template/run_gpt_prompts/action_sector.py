from persona.prompt_template.run_gpt_prompts._common import *
import random

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v1/action_location_sector_v1.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def create_prompt_input(action_description, persona, maze, test_input=None):
  act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"

  prompt_input = []

  prompt_input += [persona.scratch.get_str_name()]
  prompt_input += [persona.scratch.living_area.split(":")[1]]
  x = f"{act_world}:{persona.scratch.living_area.split(':')[1]}"
  prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]


  prompt_input += [persona.scratch.get_str_name()]
  prompt_input += [f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"]
  x = f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"
  prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]

  if persona.scratch.get_str_daily_plan_req() != "":
    prompt_input += [f"\n{persona.scratch.get_str_daily_plan_req()}"]
  else:
    prompt_input += [""]


  # MAR 11 TEMP
  accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
  curr = accessible_sector_str.split(", ")
  fin_accessible_sectors = []
  for i in curr:
    if "'s house" in i:
      if persona.scratch.last_name in i:
        fin_accessible_sectors += [i]
    else:
      fin_accessible_sectors += [i]
  accessible_sector_str = ", ".join(fin_accessible_sectors)
  # END MAR 11 TEMP

  prompt_input += [accessible_sector_str]



  action_description_1 = action_description
  action_description_2 = action_description
  if "(" in action_description:
    action_description_1 = action_description.split("(")[0].strip()
    action_description_2 = action_description.split("(")[-1][:-1]
  prompt_input += [persona.scratch.get_str_name()]
  prompt_input += [action_description_1]

  prompt_input += [action_description_2]
  prompt_input += [persona.scratch.get_str_name()]
  return prompt_input


def clean_up(gpt_response, prompt=""):
  cleaned_response = gpt_response.split("}")[0]
  return cleaned_response

def validate(gpt_response, prompt=""):
  if len(gpt_response.strip()) < 1:
    return False
  if "}" not in gpt_response:
    return False
  if "," in gpt_response:
    return False
  return True

def fail_safe():
  fs = ("kitchen")
  return fs


def run_gpt_prompt_action_sector(action_description,
                                persona,
                                maze,
                                test_input=None,
                                verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(action_description, persona, maze)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)
  y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
  x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
  if output not in x:
    # output = random.choice(x)
    output = persona.scratch.living_area.split(":")[1]

  print ("DEBUG", random.choice(x), "------", output)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
