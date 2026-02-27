from persona.prompt_template.run_gpt_prompts._common import *


def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False):
  def create_prompt_input(act_game_object, act_desp, persona):
    prompt_input = [act_game_object,
                    persona.name,
                    act_desp,
                    act_game_object,
                    act_game_object]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    if cr[-1] == ".": cr = cr[:-1]
    return cr

  def __func_validate(gpt_response, prompt=""):
    try:
      gpt_response = __func_clean_up(gpt_response, prompt="")
    except:
      return False
    return True

  def get_fail_safe(act_game_object):
    fs = f"{act_game_object} is idle"
    return fs

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    cr = gpt_response.strip()
    if cr[-1] == ".": cr = cr[:-1]
    return cr

  def __chat_func_validate(gpt_response, prompt=""): ############
    try:
      gpt_response = __func_clean_up(gpt_response, prompt="")
    except:
      return False
    return True

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 6") ########
  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt" ########
  prompt_input = create_prompt_input(act_game_object, act_desp, persona)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "being fixed" ########
  special_instruction = "The output should ONLY contain the phrase that should go in <fill in>." ########
  fail_safe = get_fail_safe(act_game_object) ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  # ChatGPT Plugin ===========================================================
