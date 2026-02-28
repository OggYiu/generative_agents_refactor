from persona.prompt_template.run_gpt_prompts._common import *


def chat_clean_up(gpt_response, prompt=""):
  gpt_response = json.loads(gpt_response)
  return gpt_response["output"]

def chat_validate(gpt_response, prompt=""):
  try:
    fields = ["output"]
    response = json.loads(gpt_response)
    for field in fields:
      if field not in response:
        return False
    return True
  except:
    return False

def fail_safe():
  return None


def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False):
  def create_prompt_input(comment, test_input=None):
    prompt_input = [comment]
    return prompt_input

  print ("11")
  prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt"
  prompt_input = create_prompt_input(comment)
  print ("22")
  prompt = generate_prompt(prompt_input, prompt_template)
  print (prompt)
  fail_safe_val = fail_safe()
  output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe_val,
                        chat_validate, chat_clean_up, verbose)
  print (output)

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
