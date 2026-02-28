from persona.prompt_template.run_gpt_prompts._common import *


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


def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None, verbose=False):
  def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None):
    prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently,
                    statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 17") ########
  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe_val = fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
