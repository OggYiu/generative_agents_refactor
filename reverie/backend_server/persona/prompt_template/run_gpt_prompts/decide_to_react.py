from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 20,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v2/decide_to_react_v1.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def create_prompt_input(init_persona, target_persona, retrieved,
                        test_input=None):



  context = ""
  for c_node in retrieved["events"]:
    curr_desc = c_node.description.split(" ")
    curr_desc[2:3] = ["was"]
    curr_desc = " ".join(curr_desc)
    context +=  f"{curr_desc}. "
  context += "\n"
  for c_node in retrieved["thoughts"]:
    context +=  f"{c_node.description}. "

  curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
  init_act_desc = init_persona.scratch.act_description
  if "(" in init_act_desc:
    init_act_desc = init_act_desc.split("(")[-1][:-1]
  if len(init_persona.scratch.planned_path) == 0:
    loc = ""
    if ":" in init_persona.scratch.act_address:
      loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
    init_p_desc = f"{init_persona.name} is already {init_act_desc} at {loc}"
  else:
    loc = ""
    if ":" in init_persona.scratch.act_address:
      loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
    init_p_desc = f"{init_persona.name} is on the way to {init_act_desc} at {loc}"

  target_act_desc = target_persona.scratch.act_description
  if "(" in target_act_desc:
    target_act_desc = target_act_desc.split("(")[-1][:-1]
  if len(target_persona.scratch.planned_path) == 0:
    loc = ""
    if ":" in target_persona.scratch.act_address:
      loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
    target_p_desc = f"{target_persona.name} is already {target_act_desc} at {loc}"
  else:
    loc = ""
    if ":" in target_persona.scratch.act_address:
      loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
    target_p_desc = f"{target_persona.name} is on the way to {target_act_desc} at {loc}"

  prompt_input = []
  prompt_input += [context]
  prompt_input += [curr_time]
  prompt_input += [init_p_desc]
  prompt_input += [target_p_desc]

  prompt_input += [init_persona.name]
  prompt_input += [init_act_desc]
  prompt_input += [target_persona.name]
  prompt_input += [target_act_desc]

  prompt_input += [init_act_desc]
  return prompt_input


def validate(gpt_response, prompt=""):
  try:
    if gpt_response.split("Answer: Option")[-1].strip().lower() in ["3", "2", "1"]:
      return True
    return False
  except:
    return False

def clean_up(gpt_response, prompt=""):
  return gpt_response.split("Answer: Option")[-1].strip().lower()

def fail_safe():
  fs = "3"
  return fs


def run_gpt_prompt_decide_to_react(persona, target_persona, retrieved,test_input=None,
                                       verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                     test_input)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
