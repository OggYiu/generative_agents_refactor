from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt" ########
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = "\U0001f6c1\U0001f9d6\u200d\u2640\ufe0f" ########
SPECIAL_INSTRUCTION = "The value for the output must ONLY contain the emojis." ########


def create_prompt_input(action_description):
  if "(" in action_description:
    action_description = action_description.split("(")[-1].split(")")[0]
  prompt_input = [action_description]
  return prompt_input


def clean_up(gpt_response, prompt=""):
  cr = gpt_response.strip()
  if len(cr) > 3:
    cr = cr[:3]
  return cr

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt="")
    if len(gpt_response) == 0:
      return False
  except: return False
  return True

def fail_safe():
  fs = "\U0001f60b"
  return fs

def chat_clean_up(gpt_response, prompt=""): ############
  cr = gpt_response.strip()
  if len(cr) > 3:
    cr = cr[:3]
  return cr

def chat_validate(gpt_response, prompt=""): ############
  try:
    clean_up(gpt_response, prompt="")
    if len(gpt_response) == 0:
      return False
  except: return False
  return True
  return True


def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, ChatGPT_safe_generate_response

  prompt_input = create_prompt_input(action_description)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe()
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
