from persona.prompt_template.run_gpt_prompts._common import *


def clean_up(gpt_response, prompt=""):
  # print ("???")
  # print (gpt_response)


  gpt_response = (prompt + gpt_response).split("What would they talk about now?")[-1].strip()
  content = re.findall('"([^"]*)"', gpt_response)

  speaker_order = []
  for i in gpt_response.split("\n"):
    name = i.split(":")[0].strip()
    if name:
      speaker_order += [name]

  ret = []
  for count, speaker in enumerate(speaker_order):
    ret += [[speaker, content[count]]]

  return ret

def validate(gpt_response, prompt=""):
  try:
    clean_up(gpt_response, prompt)
    return True
  except:
    return False

def fail_safe(init_persona, target_persona):
  convo = [[init_persona.name, "Hi!"],
           [target_persona.name, "Hi!"]]
  return convo


def run_gpt_prompt_create_conversation(persona, target_persona, curr_loc,
                                       test_input=None, verbose=False):
  def create_prompt_input(init_persona, target_persona, curr_loc,
                          test_input=None):

    prev_convo_insert = "\n"
    if init_persona.a_mem.seq_chat:
      for i in init_persona.a_mem.seq_chat:
        if i.object == target_persona.scratch.name:
          v1 = int((init_persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation.\n'
          for row in i.filling:
            prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
          break
    if prev_convo_insert == "\n":
      prev_convo_insert = ""
    if init_persona.a_mem.seq_chat:
      if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480:
        prev_convo_insert = ""


    init_persona_thought_nodes = init_persona.a_mem.retrieve_relevant_thoughts(target_persona.scratch.act_event[0],
                                target_persona.scratch.act_event[1],
                                target_persona.scratch.act_event[2])
    init_persona_thought = ""
    for i in init_persona_thought_nodes:
      init_persona_thought += f"-- {i.description}\n"

    target_persona_thought_nodes = target_persona.a_mem.retrieve_relevant_thoughts(init_persona.scratch.act_event[0],
                                init_persona.scratch.act_event[1],
                                init_persona.scratch.act_event[2])
    target_persona_thought = ""
    for i in target_persona_thought_nodes:
      target_persona_thought += f"-- {i.description}\n"

    init_persona_curr_desc = ""
    if init_persona.scratch.planned_path:
      init_persona_curr_desc = f"{init_persona.name} is on the way to {init_persona.scratch.act_description}"
    else:
      init_persona_curr_desc = f"{init_persona.name} is {init_persona.scratch.act_description}"

    target_persona_curr_desc = ""
    if target_persona.scratch.planned_path:
      target_persona_curr_desc = f"{target_persona.name} is on the way to {target_persona.scratch.act_description}"
    else:
      target_persona_curr_desc = f"{target_persona.name} is {target_persona.scratch.act_description}"


    curr_loc = curr_loc["arena"]

    prompt_input = []
    prompt_input += [init_persona.scratch.get_str_iss()]
    prompt_input += [target_persona.scratch.get_str_iss()]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [init_persona_thought]

    prompt_input += [target_persona.name]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona_thought]

    prompt_input += [init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S")]

    prompt_input += [init_persona_curr_desc]
    prompt_input += [target_persona_curr_desc]

    prompt_input += [prev_convo_insert]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]

    prompt_input += [curr_loc]
    prompt_input += [init_persona.name]
    return prompt_input


  gpt_param = {"engine": "text-davinci-003", "max_tokens": 1000,
               "temperature": 0.7, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/create_conversation_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, curr_loc,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe_val = fail_safe(persona, target_persona)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe_val,
                                   validate, clean_up)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe_val]
