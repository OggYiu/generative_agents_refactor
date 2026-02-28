from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 500,
             "temperature": 1, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v2/daily_planning_v6.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def create_prompt_input(persona, wake_up_hour, test_input=None):
  if test_input: return test_input
  prompt_input = []
  prompt_input += [persona.scratch.get_str_iss()]
  prompt_input += [persona.scratch.get_str_lifestyle()]
  prompt_input += [persona.scratch.get_str_curr_date_str()]
  prompt_input += [persona.scratch.get_str_firstname()]
  prompt_input += [f"{str(wake_up_hour)}:00 am"]
  return prompt_input


def clean_up(gpt_response, prompt=""):
  cr = []
  _cr = gpt_response.split(")")
  for i in _cr:
    if i[-1].isdigit():
      i = i[:-1].strip()
      if i[-1] == "." or i[-1] == ",":
        cr += [i[:-1].strip()]
  return cr

def validate(gpt_response, prompt=""):
  try: clean_up(gpt_response, prompt="")
  except:
    return False
  return True

def fail_safe():
  fs = ['wake up and complete the morning routine at 6:00 am',
        'eat breakfast at 7:00 am',
        'read a book from 8:00 am to 12:00 pm',
        'have lunch at 12:00 pm',
        'take a nap from 1:00 pm to 4:00 pm',
        'relax and watch TV from 7:00 pm to 8:00 pm',
        'go to bed at 11:00 pm']
  return fs


def run_gpt_prompt_daily_plan(persona,
                              wake_up_hour,
                              test_input=None,
                              verbose=False):
  """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form:
  'wake up and complete the morning routine at 6:00 am',
  'eat breakfast at 7:00 am',..
  Note that the actions come without a period.

  INPUT:
    persona: The Persona class instance
  OUTPUT:
    a list of daily actions in broad strokes.
  """
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe()

  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)
  output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
              + output)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
