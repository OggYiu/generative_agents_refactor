from persona.prompt_template.run_gpt_prompts._common import *
from persona.prompt_template.run_gpt_prompts._helpers import get_random_alphanumeric

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 50,
             "temperature": 0.5, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
PROMPT_TEMPLATE = "persona/prompt_template/v2/generate_hourly_schedule_v3.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def create_prompt_input(persona,
                        curr_hour_str,
                        p_f_ds_hourly_org,
                        hour_str,
                        intermission2=None,
                        test_input=None):
  if test_input: return test_input
  schedule_format = ""
  for i in hour_str:
    schedule_format += f"[{persona.scratch.get_str_curr_date_str()} -- {i}]"
    schedule_format += f" Activity: [Fill in]\n"
  schedule_format = schedule_format[:-1]

  intermission_str = f"Here the originally intended hourly breakdown of"
  intermission_str += f" {persona.scratch.get_str_firstname()}'s schedule today: "
  for count, i in enumerate(persona.scratch.daily_req):
    intermission_str += f"{str(count+1)}) {i}, "
  intermission_str = intermission_str[:-2]

  prior_schedule = ""
  if p_f_ds_hourly_org:
    prior_schedule = "\n"
    for count, i in enumerate(p_f_ds_hourly_org):
      prior_schedule += f"[(ID:{get_random_alphanumeric()})"
      prior_schedule += f" {persona.scratch.get_str_curr_date_str()} --"
      prior_schedule += f" {hour_str[count]}] Activity:"
      prior_schedule += f" {persona.scratch.get_str_firstname()}"
      prior_schedule += f" is {i}\n"

  prompt_ending = f"[(ID:{get_random_alphanumeric()})"
  prompt_ending += f" {persona.scratch.get_str_curr_date_str()}"
  prompt_ending += f" -- {curr_hour_str}] Activity:"
  prompt_ending += f" {persona.scratch.get_str_firstname()} is"

  if intermission2:
    intermission2 = f"\n{intermission2}"

  prompt_input = []
  prompt_input += [schedule_format]
  prompt_input += [persona.scratch.get_str_iss()]

  prompt_input += [prior_schedule + "\n"]
  prompt_input += [intermission_str]
  if intermission2:
    prompt_input += [intermission2]
  else:
    prompt_input += [""]
  prompt_input += [prompt_ending]

  return prompt_input


def clean_up(gpt_response, prompt=""):
  cr = gpt_response.strip()
  if cr[-1] == ".":
    cr = cr[:-1]
  return cr

def validate(gpt_response, prompt=""):
  try: clean_up(gpt_response, prompt="")
  except: return False
  return True

def fail_safe():
  fs = "asleep"
  return fs


def run_gpt_prompt_generate_hourly_schedule(persona,
                                            curr_hour_str,
                                            p_f_ds_hourly_org,
                                            hour_str,
                                            intermission2=None,
                                            test_input=None,
                                            verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(persona,
                                     curr_hour_str,
                                     p_f_ds_hourly_org,
                                     hour_str,
                                     intermission2,
                                     test_input)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe()

  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
