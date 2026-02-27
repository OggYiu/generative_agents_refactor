from persona.prompt_template.run_gpt_prompts._common import *


def run_gpt_prompt_event_triple(action_description, persona, verbose=False):
  def create_prompt_input(action_description, persona):
    if "(" in action_description:
      action_description = action_description.split("(")[-1].split(")")[0]
    prompt_input = [persona.name,
                    action_description,
                    persona.name]
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

  def get_fail_safe(persona):
    fs = (persona.name, "is", "idle")
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 30,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
  prompt_input = create_prompt_input(action_description, persona)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(persona) ########
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  output = (persona.name, output[0], output[1])

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
