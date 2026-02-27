from persona.prompt_template.run_gpt_prompts._common import *


def run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False):
  def create_prompt_input(act_game_object, act_obj_desc):
    prompt_input = [act_game_object,
                    act_obj_desc,
                    act_game_object]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    cr = [i.strip() for i in cr.split(")")[0].split(",")]
    return cr

  def __func_validate(gpt_response, prompt=""):
    try:
      gpt_response = __func_clean_up(gpt_response, prompt="")
      if len(gpt_response) != 2:
        return False
    except: return False
    return True

  def get_fail_safe(act_game_object):
    fs = (act_game_object, "is", "idle")
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 30,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
  prompt_input = create_prompt_input(act_game_object, act_obj_desc)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(act_game_object)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  output = (act_game_object, output[0], output[1])

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
