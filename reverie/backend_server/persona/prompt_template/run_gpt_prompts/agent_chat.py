from persona.prompt_template.run_gpt_prompts._common import *


def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                               curr_context,
                               init_summ_idea,
                               target_summ_idea, test_input=None, verbose=False):
  def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None):
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

  def __func_clean_up(gpt_response, prompt=""):
    print (gpt_response)

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



  def __func_validate(gpt_response, prompt=""):
    try:
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False

  def get_fail_safe():
    return "..."




  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    # ret = ast.literal_eval(gpt_response)

    print ("a;dnfdap98fh4p9enf HEREE!!!")
    for row in gpt_response:
      print (row)

    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    return True


  # print ("HERE JULY 23 -- ----- ") ########
  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]' ########
  special_instruction = 'The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"].' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  # print ("HERE END JULY 23 -- ----- ") ########
  if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  # ChatGPT Plugin ===========================================================
