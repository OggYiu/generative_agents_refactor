from persona.prompt_template.run_gpt_prompts._common import *


def clean_up(gpt_response, prompt=""):
  new_schedule = prompt + " " + gpt_response.strip()
  new_schedule = new_schedule.split("The revised schedule:")[-1].strip()
  new_schedule = new_schedule.split("\n")

  ret_temp = []
  for i in new_schedule:
    ret_temp += [i.split(" -- ")]

  ret = []
  for time_str, action in ret_temp:
    start_time = time_str.split(" ~ ")[0].strip()
    end_time = time_str.split(" ~ ")[1].strip()
    delta = datetime.datetime.strptime(end_time, "%H:%M") - datetime.datetime.strptime(start_time, "%H:%M")
    delta_min = int(delta.total_seconds()/60)
    if delta_min < 0: delta_min = 0
    ret += [[action, delta_min]]

  return ret

def validate(gpt_response, prompt=""):
  try:
    gpt_response = clean_up(gpt_response, prompt)
    dur_sum = 0
    for act, dur in gpt_response:
      dur_sum += dur
      if str(type(act)) != "<class 'str'>":
        return False
      if str(type(dur)) != "<class 'int'>":
        return False
    x = prompt.split("\n")[0].split("originally planned schedule from")[-1].strip()[:-1]
    x = [datetime.datetime.strptime(i.strip(), "%H:%M %p") for i in x.split(" to ")]
    delta_min = int((x[1] - x[0]).total_seconds()/60)

    if int(dur_sum) != int(delta_min):
      return False

  except:
    return False
  return True

def fail_safe(main_act_dur, truncated_act_dur):
  dur_sum = 0
  for act, dur in main_act_dur: dur_sum += dur

  ret = truncated_act_dur[:]
  ret += main_act_dur[len(ret)-1:]

  # If there are access, we need to trim...
  ret_dur_sum = 0
  count = 0
  over = None
  for act, dur in ret:
    ret_dur_sum += dur
    if ret_dur_sum == dur_sum:
      break
    if ret_dur_sum > dur_sum:
      over = ret_dur_sum - dur_sum
      break
    count += 1

  if over:
    ret = ret[:count+1]
    ret[-1][1] -= over

  return ret


def run_gpt_prompt_new_decomp_schedule(persona,
                                       main_act_dur,
                                       truncated_act_dur,
                                       start_time_hour,
                                       end_time_hour,
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input=None,
                                       verbose=False):
  def create_prompt_input(persona,
                           main_act_dur,
                           truncated_act_dur,
                           start_time_hour,
                           end_time_hour,
                           inserted_act,
                           inserted_act_dur,
                           test_input=None):
    persona_name = persona.name
    start_hour_str = start_time_hour.strftime("%H:%M %p")
    end_hour_str = end_time_hour.strftime("%H:%M %p")

    original_plan = ""
    for_time = start_time_hour
    for i in main_act_dur:
      original_plan += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      original_plan += "\n"
      for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init = ""
    for_time = start_time_hour
    for count, i in enumerate(truncated_act_dur):
      new_plan_init += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      new_plan_init += "\n"
      if count < len(truncated_act_dur) - 1:
        for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M") + " ~"

    prompt_input = [persona_name,
                    start_hour_str,
                    end_hour_str,
                    original_plan,
                    persona_name,
                    inserted_act,
                    inserted_act_dur,
                    persona_name,
                    start_hour_str,
                    end_hour_str,
                    end_hour_str,
                    new_plan_init]
    return prompt_input

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 1000,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/new_decomp_schedule_v1.txt"
  prompt_input = create_prompt_input(persona,
                                     main_act_dur,
                                     truncated_act_dur,
                                     start_time_hour,
                                     end_time_hour,
                                     inserted_act,
                                     inserted_act_dur,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe_val = fail_safe(main_act_dur, truncated_act_dur)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe_val,
                                   validate, clean_up)

  # print ("* * * * output")
  # print (output)
  # print ('* * * * fail_safe')
  # print (fail_safe)



  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
