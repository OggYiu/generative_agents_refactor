from persona.prompt_template.run_gpt_prompts._common import *
from persona.prompt_template.run_gpt_prompts._helpers import extract_first_json_dict

GPT_PARAM = {"engine": "text-davinci-003", "max_tokens": 50,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt"
REPEAT = 3
LLM_CALL_TYPE = "chat_old"


def chat_clean_up(gpt_response, prompt=""):
  gpt_response = extract_first_json_dict(gpt_response)

  cleaned_dict = dict()
  cleaned = []
  for key, val in gpt_response.items():
    cleaned += [val]
  cleaned_dict["utterance"] = cleaned[0]
  cleaned_dict["end"] = True
  if "f" in str(cleaned[1]) or "F" in str(cleaned[1]):
    cleaned_dict["end"] = False

  return cleaned_dict

def chat_validate(gpt_response, prompt=""):
  print ("ugh...")
  try:
    # print ("debug 1")
    # print (gpt_response)
    # print ("debug 2")

    print (extract_first_json_dict(gpt_response))
    # print ("debug 3")

    return True
  except:
    return False

def fail_safe():
  cleaned_dict = dict()
  cleaned_dict["utterance"] = "..."
  cleaned_dict["end"] = False
  return cleaned_dict


def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
  persona = init_persona
  prev_convo_insert = "\n"
  if persona.a_mem.seq_chat:
    for i in persona.a_mem.seq_chat:
      if i.object == target_persona.scratch.name:
        v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
        prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
        break
  if prev_convo_insert == "\n":
    prev_convo_insert = ""
  if persona.a_mem.seq_chat:
    if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480:
      prev_convo_insert = ""
  print (prev_convo_insert)

  curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
  curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
  curr_location = f"{curr_arena} in {curr_sector}"

  retrieved_str = ""
  for key, vals in retrieved.items():
    for v in vals:
      retrieved_str += f"- {v.description}\n"


  convo_str = ""
  for i in curr_chat:
    convo_str += ": ".join(i) + "\n"
  if convo_str == "":
    convo_str = "[The conversation has not started yet -- start it!]"

  init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
  prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
    curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
    convo_str, init_persona.scratch.name, target_persona.scratch.name,
    init_persona.scratch.name, init_persona.scratch.name,
    init_persona.scratch.name
    ]
  return prompt_input


def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import (
    ChatGPT_safe_generate_response_OLD,
    generate_prompt,
  )

  print ("11")
  prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)
  print ("22")
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  print (prompt)
  fail_safe_val = fail_safe()
  output = ChatGPT_safe_generate_response_OLD(prompt, REPEAT, fail_safe_val,
                        chat_validate, chat_clean_up, verbose)
  print (output)

  return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
