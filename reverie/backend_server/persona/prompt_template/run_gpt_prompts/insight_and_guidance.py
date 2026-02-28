from persona.prompt_template.run_gpt_prompts._common import *


def clean_up(gpt_response, prompt=""):
  gpt_response = "1. " + gpt_response.strip()
  ret = dict()
  for i in gpt_response.split("\n"):
    row = i.split(". ")[-1]
    thought = row.split("(because of ")[0].strip()
    evi_raw = row.split("(because of ")[1].split(")")[0].strip()
    evi_raw = re.findall(r'\d+', evi_raw)
    evi_raw = [int(i.strip()) for i in evi_raw]
    ret[thought] = evi_raw
  return ret

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False

def fail_safe(n):
  return ["I am hungry"] * n


def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False):
  def create_prompt_input(persona, statements, n, test_input=None):
    prompt_input = [statements, str(n)]
    return prompt_input




  gpt_param = {"engine": "text-davinci-003", "max_tokens": 150,
               "temperature": 0.5, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe_val = fail_safe(n)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
