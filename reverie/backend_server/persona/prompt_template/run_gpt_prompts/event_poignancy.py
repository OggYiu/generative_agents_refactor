from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt" ########
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = "5" ########
SPECIAL_INSTRUCTION = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########


def create_prompt_input(persona, event_description, test_input=None):
  prompt_input = [persona.scratch.name,
                  persona.scratch.get_str_iss(),
                  persona.scratch.name,
                  event_description]
  return prompt_input


def clean_up(gpt_response, prompt=""):
  gpt_response = int(gpt_response.strip())
  return gpt_response

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False

def fail_safe():
  return 4

def chat_clean_up(gpt_response, prompt=""):
  gpt_response = int(gpt_response)
  return gpt_response

def chat_validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False


def run_gpt_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, ChatGPT_safe_generate_response

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 7") ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
