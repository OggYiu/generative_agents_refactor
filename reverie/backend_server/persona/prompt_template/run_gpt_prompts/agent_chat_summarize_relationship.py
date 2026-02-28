from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt"
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = 'Jane Doe is working on a project'
SPECIAL_INSTRUCTION = 'The output should be a string that responds to the question.'


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

def chat_clean_up(gpt_response, prompt=""):
  return gpt_response.split('"')[0].strip()

def chat_validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False


def create_prompt_input(persona, target_persona, statements, test_input=None):
  prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
  return prompt_input


def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, ChatGPT_safe_generate_response

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 18") ########
  prompt_input = create_prompt_input(persona, target_persona, statements)  ########
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
