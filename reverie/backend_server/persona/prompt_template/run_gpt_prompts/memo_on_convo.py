from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt"
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = 'Jane Doe was interesting to talk to.'
SPECIAL_INSTRUCTION = 'The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed'

GPT_PARAM_FALLBACK = {"engine": "text-davinci-003", "max_tokens": 50,
                      "temperature": 0, "top_p": 1, "stream": False,
                      "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE_FALLBACK = "persona/prompt_template/v2/memo_on_convo_v1.txt"
REPEAT_FALLBACK = 5


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

def chat_clean_up(gpt_response, prompt=""): ############
  return gpt_response.strip()

def chat_validate(gpt_response, prompt=""): ############
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False


def create_prompt_input(persona, all_utt, test_input=None):
  prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
  return prompt_input


def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import (
    ChatGPT_safe_generate_response,
    safe_generate_response,
    generate_prompt,
  )
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================

  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE_FALLBACK)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, GPT_PARAM_FALLBACK, REPEAT_FALLBACK, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE_FALLBACK, persona, GPT_PARAM_FALLBACK,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM_FALLBACK, prompt_input, fail_safe_val]
