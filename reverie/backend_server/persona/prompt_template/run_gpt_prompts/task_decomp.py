from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 1000,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v2/task_decomp_v4.txt"
REPEAT = 5
LLM_CALL_TYPE = "completion"


def create_prompt_input(persona, task, duration, test_input=None):

  """
  Today is Saturday June 25. From 00:00 ~ 06:00am, Maeve is
  planning on sleeping, 06:00 ~ 07:00am, Maeve is
  planning on waking up and doing her morning routine,
  and from 07:00am ~08:00am, Maeve is planning on having breakfast.
  """

  curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
  all_indices = []
  # if curr_f_org_index > 0:
  #   all_indices += [curr_f_org_index-1]
  all_indices += [curr_f_org_index]
  if curr_f_org_index+1 <= len(persona.scratch.f_daily_schedule_hourly_org):
    all_indices += [curr_f_org_index+1]
  if curr_f_org_index+2 <= len(persona.scratch.f_daily_schedule_hourly_org):
    all_indices += [curr_f_org_index+2]

  curr_time_range = ""

  summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
  summ_str += f'From '
  for index in all_indices:
    if index < len(persona.scratch.f_daily_schedule_hourly_org):
      start_min = 0
      for i in range(index):
        start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
      end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
      start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                    + datetime.timedelta(minutes=start_min))
      end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                    + datetime.timedelta(minutes=end_min))
      start_time_str = start_time.strftime("%H:%M%p")
      end_time_str = end_time.strftime("%H:%M%p")
      summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
      if curr_f_org_index+1 == index:
        curr_time_range = f'{start_time_str} ~ {end_time_str}'
  summ_str = summ_str[:-2] + "."

  prompt_input = []
  prompt_input += [persona.scratch.get_str_iss()]
  prompt_input += [summ_str]
  # prompt_input += [persona.scratch.get_str_curr_date_str()]
  prompt_input += [persona.scratch.get_str_firstname()]
  prompt_input += [persona.scratch.get_str_firstname()]
  prompt_input += [task]
  prompt_input += [curr_time_range]
  prompt_input += [duration]
  prompt_input += [persona.scratch.get_str_firstname()]
  return prompt_input


def clean_up(gpt_response, prompt=""):
  # v4: all items include "N) Name is" prefix, strip first 3 words uniformly
  temp = [i.strip() for i in gpt_response.split("\n") if i.strip()]
  _cr = []
  cr = []
  for i in temp:
    _cr += [" ".join([j.strip() for j in i.split(" ")][3:])]
  for count, i in enumerate(_cr):
    k = [j.strip() for j in i.split("(duration in minutes:")]
    task = k[0]
    if task[-1] == ".":
      task = task[:-1]
    duration = int(k[1].split(",")[0].strip())
    cr += [[task, duration]]

  total_expected_min = int(prompt.split("(total duration in minutes")[-1]
                                 .strip().split(")")[0].strip())

  # TODO -- now, you need to make sure that this is the same as the sum of
  #         the current action sequence.
  curr_min_slot = [["dummy", -1],] # (task_name, task_index)
  for count, i in enumerate(cr):
    i_task = i[0]
    i_duration = i[1]

    i_duration -= (i_duration % 5)
    if i_duration > 0:
      for j in range(i_duration):
        curr_min_slot += [(i_task, count)]
  curr_min_slot = curr_min_slot[1:]

  if len(curr_min_slot) > total_expected_min:
    curr_min_slot = curr_min_slot[:total_expected_min]
  elif len(curr_min_slot) < total_expected_min:
    last_task = curr_min_slot[-1]
    for i in range(total_expected_min - len(curr_min_slot)):
      curr_min_slot += [last_task]

  cr_ret = [["dummy", -1],]
  for task, task_index in curr_min_slot:
    if task != cr_ret[-1][0]:
      cr_ret += [[task, 1]]
    else:
      cr_ret[-1][1] += 1
  cr = cr_ret[1:]

  return cr

def validate(gpt_response, prompt=""):
  try:
    result = clean_up(gpt_response, prompt)
    return len(result) > 0
  except:
    return False

def fail_safe():
  fs = ["asleep"]
  return fs


def run_gpt_prompt_task_decomp(persona,
                               task,
                               duration,
                               test_input=None,
                               verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, safe_generate_response
  from persona.prompt_template.print_prompt import print_run_prompts
  from utils import debug

  prompt_input = create_prompt_input(persona, task, duration)
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe()

  output = safe_generate_response(prompt, GPT_PARAM, REPEAT, fail_safe(),
                                   validate, clean_up)

  # Clamp output to the requested duration to prevent overflows
  fin_output = []
  time_sum = 0
  for i_task, i_duration in output:
    time_sum += i_duration
    if time_sum <= duration:
      fin_output += [[i_task, i_duration]]
    else:
      break
  ftime_sum = 0
  for fi_task, fi_duration in fin_output:
    ftime_sum += fi_duration
  fin_output[-1][1] += (duration - ftime_sum)
  output = fin_output



  task_decomp = output
  ret = []
  for decomp_task, duration in task_decomp:
    ret += [[f"{task} ({decomp_task})", duration]]
  output = ret


  if debug or verbose:
    print_run_prompts(PROMPT_TEMPLATE, persona, GPT_PARAM,
                      prompt_input, prompt, output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
