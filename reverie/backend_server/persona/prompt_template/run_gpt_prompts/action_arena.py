from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v1/action_location_object_vMar11.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


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


def create_prompt_input(action_description, persona, maze, act_world, act_sector, test_input=None):
  prompt_input = []
  # prompt_input += [persona.scratch.get_str_name()]
  # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
  # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["sector"]]
  prompt_input += [persona.scratch.get_str_name()]
  x = f"{act_world}:{act_sector}"
  prompt_input += [act_sector]

  # MAR 11 TEMP
  accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
  curr = accessible_arena_str.split(", ")
  fin_accessible_arenas = []
  for i in curr:
    if "'s room" in i:
      if persona.scratch.last_name in i:
        fin_accessible_arenas += [i]
    else:
      fin_accessible_arenas += [i]
  accessible_arena_str = ", ".join(fin_accessible_arenas)
  # END MAR 11 TEMP


  prompt_input += [accessible_arena_str]


  action_description_1 = action_description
  action_description_2 = action_description
  if "(" in action_description:
    action_description_1 = action_description.split("(")[0].strip()
    action_description_2 = action_description.split("(")[-1][:-1]
  prompt_input += [persona.scratch.get_str_name()]
  prompt_input += [action_description_1]

  prompt_input += [action_description_2]
  prompt_input += [persona.scratch.get_str_name()]



  prompt_input += [act_sector]

  prompt_input += [accessible_arena_str]
  # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
  # x = f"{maze.access_tile(persona.scratch.curr_tile)['world']}:{maze.access_tile(persona.scratch.curr_tile)['sector']}:{maze.access_tile(persona.scratch.curr_tile)['arena']}"
  # prompt_input += [persona.s_mem.get_str_accessible_arena_game_objects(x)]


  return prompt_input


def run_gpt_prompt_action_arena(action_description,
                                persona,
                                maze, act_world, act_sector,
                                test_input=None,
                                verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(action_description, persona, maze, act_world, act_sector)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)
  print (output)
  # y = f"{act_world}:{act_sector}"
  # x = [i.strip() for i in persona.s_mem.get_str_accessible_sector_arenas(y).split(",")]
  # if output not in x:
  #   output = random.choice(x)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
