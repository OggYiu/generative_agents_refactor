from persona.prompt_template.run_gpt_prompts._common import *


def clean_up(gpt_response, prompt=""):
  print ("???")
  print (gpt_response)
  gpt_response = gpt_response.strip().split("Emotive keywords:")
  factual = [i.strip() for i in gpt_response[0].split(",")]
  emotive = [i.strip() for i in gpt_response[1].split(",")]
  all_keywords = factual + emotive
  ret = []
  for i in all_keywords:
    if i:
      i = i.lower()
      if i[-1] == ".":
        i = i[:-1]
      ret += [i]
  print (ret)
  return set(ret)

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False

def fail_safe():
  return []


def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False):
  def create_prompt_input(description, test_input=None):
    if "\n" in description:
      description = description.replace("\n", " <LINE_BREAK> ")
    prompt_input = [description]
    return prompt_input

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
  prompt_input = create_prompt_input(description, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe_val = fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe_val,
                                   validate, clean_up)


  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
