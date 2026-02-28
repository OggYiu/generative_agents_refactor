from persona.prompt_template.run_gpt_prompts._common import *


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
  def create_prompt_input(action_description):
    if "(" in action_description:
      action_description = action_description.split("(")[-1].split(")")[0]
    prompt_input = [action_description]
    return prompt_input

  # ChatGPT Plugin ===========================================================
  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 4") ########
  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt" ########
  prompt_input = create_prompt_input(action_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "\U0001f6c1\U0001f9d6\u200d\u2640\ufe0f" ########
  special_instruction = "The value for the output must ONLY contain the emojis." ########
  fail_safe_val = fail_safe()
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
