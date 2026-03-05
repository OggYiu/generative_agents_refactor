from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt"
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]'
SPECIAL_INSTRUCTION = "Output must be a list of str."

GPT_PARAM_FALLBACK = {"engine": "text-davinci-003", "max_tokens": 150,
                      "temperature": 0, "top_p": 1, "stream": False,
                      "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE_FALLBACK = "persona/prompt_template/v2/generate_focal_pt_v1.txt"
REPEAT_FALLBACK = 5


def clean_up(gpt_response, prompt=""):
  gpt_response = "1) " + gpt_response.strip()
  ret = []
  for i in gpt_response.split("\n"):
    ret += [i.split(") ")[-1]]
  return ret

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False

def fail_safe(n):
  return ["Who am I"] * n

def chat_clean_up(gpt_response, prompt=""): ############
  ret = ast.literal_eval(gpt_response)
  return ret

def chat_validate(gpt_response, prompt=""): ############
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False


def create_prompt_input(persona, statements, n, test_input=None):
  prompt_input = [statements, str(n)]
  return prompt_input


def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import (
    ChatGPT_safe_generate_response,
    safe_generate_response,
    generate_prompt,
  )
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe(n) ########
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================






  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE_FALLBACK)

  fail_safe_val = fail_safe(n)
  output = safe_generate_response(prompt, GPT_PARAM_FALLBACK, REPEAT_FALLBACK, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE_FALLBACK, persona, GPT_PARAM_FALLBACK,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM_FALLBACK, prompt_input, fail_safe_val]
