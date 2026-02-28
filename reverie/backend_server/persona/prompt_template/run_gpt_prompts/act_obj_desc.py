from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt" ########
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = "being fixed" ########
SPECIAL_INSTRUCTION = "The output should ONLY contain the phrase that should go in <fill in>." ########


def create_prompt_input(act_game_object, act_desp, persona):
  prompt_input = [act_game_object,
                  persona.name,
                  act_desp,
                  act_game_object,
                  act_game_object]
  return prompt_input


def clean_up(gpt_response, prompt=""):
  cr = gpt_response.strip()
  if cr[-1] == ".": cr = cr[:-1]
  return cr

def validate(gpt_response, prompt=""):
  try:
    gpt_response = clean_up(gpt_response, prompt="")
  except:
    return False
  return True

def fail_safe(act_game_object):
  fs = f"{act_game_object} is idle"
  return fs

def chat_clean_up(gpt_response, prompt=""):
  cr = gpt_response.strip()
  if cr[-1] == ".": cr = cr[:-1]
  return cr

def chat_validate(gpt_response, prompt=""):
  try:
    gpt_response = clean_up(gpt_response, prompt="")
  except:
    return False
  return True


def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, ChatGPT_safe_generate_response

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 6") ########
  prompt_input = create_prompt_input(act_game_object, act_desp, persona)  ########
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe(act_game_object) ########
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
