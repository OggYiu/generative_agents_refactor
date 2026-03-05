from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt"
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = "conversing about what to eat for lunch"
SPECIAL_INSTRUCTION = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who are chatting)."


def clean_up(gpt_response, prompt=""):
  ret = "conversing about " + gpt_response.strip()
  return ret

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False

def fail_safe():
  return "conversing with a housemate about morning greetings"

def chat_clean_up(gpt_response, prompt=""):
  ret = "conversing about " + gpt_response.strip()
  return ret

def chat_validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False


def create_prompt_input(conversation, test_input=None):
  convo_str = ""
  for row in conversation:
    convo_str += f'{row[0]}: "{row[1]}"\n'

  prompt_input = [convo_str]
  return prompt_input


def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, ChatGPT_safe_generate_response

  prompt_input = create_prompt_input(conversation, test_input)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
