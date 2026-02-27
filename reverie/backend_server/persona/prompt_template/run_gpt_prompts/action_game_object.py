from persona.prompt_template.run_gpt_prompts._common import *


def run_gpt_prompt_action_game_object(action_description,
                                      persona,
                                      maze,
                                      temp_address,
                                      test_input=None,
                                      verbose=False):
  def create_prompt_input(action_description,
                          persona,
                          temp_address,
                          test_input=None):
    prompt_input = []
    if "(" in action_description:
      action_description = action_description.split("(")[-1][:-1]

    prompt_input += [action_description]
    prompt_input += [persona
                     .s_mem.get_str_accessible_arena_game_objects(temp_address)]
    return prompt_input

  def __func_validate(gpt_response, prompt=""):
    if len(gpt_response.strip()) < 1:
      return False
    return True

  def __func_clean_up(gpt_response, prompt=""):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  def get_fail_safe():
    fs = ("bed")
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 15,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v1/action_object_v2.txt"
  prompt_input = create_prompt_input(action_description,
                                     persona,
                                     temp_address,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  x = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
  if output not in x:
    output = random.choice(x)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
