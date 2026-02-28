from persona.prompt_template.run_gpt_prompts._common import *


def clean_up(gpt_response, prompt=""):
  # Extract hour from formats like "6 am", "6:00 AM", "Isabella's wake up hour: 6:00 AM"
  text = gpt_response.strip().lower()
  match = re.search(r'(\d{1,2})(?::\d{2})?\s*([ap]m)', text)
  if match:
    return int(match.group(1))
  # Fallback: grab the first number in the response
  return int(re.search(r'\d+', text).group())

def validate(gpt_response, prompt=""):
  try: clean_up(gpt_response, prompt="")
  except: return False
  return True

def fail_safe():
  fs = 8
  return fs


def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False):
  """
  Given the persona, returns an integer that indicates the hour when the
  persona wakes up.

  INPUT:
    persona: The Persona class instance
  OUTPUT:
    integer for the wake up hour.
  """
  def create_prompt_input(persona, test_input=None):
    if test_input: return test_input
    prompt_input = [persona.scratch.get_str_iss(),
                    persona.scratch.get_str_lifestyle(),
                    persona.scratch.get_str_firstname()]
    return prompt_input

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 5,
             "temperature": 0.8, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
  prompt_input = create_prompt_input(persona, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe_val = fail_safe()

  output = safe_generate_response(prompt, gpt_param, 5, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
