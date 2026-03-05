from persona.prompt_template.run_gpt_prompts._common import *

GPT_PARAM = {"engine": "text-davinci-002", "max_tokens": 15,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
PROMPT_TEMPLATE = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt"
REPEAT = 3
LLM_CALL_TYPE = "chat"
EXAMPLE_OUTPUT = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]'
SPECIAL_INSTRUCTION = 'The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"].'


def clean_up(gpt_response, prompt=""):
  gpt_response = (prompt + gpt_response).split("Here is their conversation.")[-1].strip()
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

def fail_safe():
  return "..."

def chat_clean_up(gpt_response, prompt=""):
  return gpt_response

def chat_validate(gpt_response, prompt=""):
  return True


def create_prompt_input(maze, persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None):
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
  curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
  curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
  curr_location = f"{curr_arena} in {curr_sector}"


  prompt_input = [persona.scratch.currently,
                  target_persona.scratch.currently,
                  prev_convo_insert,
                  curr_context,
                  curr_location,

                  persona.scratch.name,
                  init_summ_idea,
                  persona.scratch.name,
                  target_persona.scratch.name,

                  target_persona.scratch.name,
                  target_summ_idea,
                  target_persona.scratch.name,
                  persona.scratch.name,

                  persona.scratch.name]
  return prompt_input


def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                               curr_context,
                               init_summ_idea,
                               target_summ_idea, test_input=None, verbose=False):
  from persona.prompt_template.gpt_structure import generate_prompt, ChatGPT_safe_generate_response

  # print ("HERE JULY 23 -- ----- ") ########
  prompt_input = create_prompt_input(maze, persona, target_persona, curr_context, init_summ_idea, target_summ_idea)  ########
  prompt = generate_prompt(prompt_input, PROMPT_TEMPLATE)
  fail_safe_val = fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, EXAMPLE_OUTPUT, SPECIAL_INSTRUCTION, REPEAT, fail_safe_val,
                                          chat_validate, chat_clean_up, True)
  # print ("HERE END JULY 23 -- ----- ") ########
  if output != False:
    return output, [output, prompt, GPT_PARAM, prompt_input, fail_safe_val]
  # ChatGPT Plugin ===========================================================
