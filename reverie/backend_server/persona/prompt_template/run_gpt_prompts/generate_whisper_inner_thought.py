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


def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False):
  def create_prompt_input(persona, whisper, test_input=None):
    prompt_input = [persona.scratch.name, whisper]
    return prompt_input

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
  prompt_input = create_prompt_input(persona, whisper)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
