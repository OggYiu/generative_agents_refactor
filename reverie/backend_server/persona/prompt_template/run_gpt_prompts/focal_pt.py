from persona.prompt_template.run_gpt_prompts._common import *


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


def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False):
  def create_prompt_input(persona, statements, n, test_input=None):
    prompt_input = [statements, str(n)]
    return prompt_input


  # ChatGPT Plugin ===========================================================
  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 12") ########
  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt" ########
  prompt_input = create_prompt_input(persona, statements, n)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]' ########
  special_instruction = "Output must be a list of str." ########
  fail_safe_val = fail_safe(n) ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================






  gpt_param = {"engine": "text-davinci-003", "max_tokens": 150,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/generate_focal_pt_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe_val = fail_safe(n)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
