from persona.prompt_template.run_gpt_prompts.run_gpt_prompts_base import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *

from utils import *

class FirstDailyPlan(RunGptPromptsBase):
    """
    Generates the daily plan for the persona. 
    Basically the long term planning that spans a day. Returns a list of actions
    that the persona will take today. Usually comes in the following form: 
    'wake up and complete the morning routine at 6:00 am', 
    'eat breakfast at 7:00 am',.. 
    Note that the actions come without a period. 

    Persona state: identity stable set, lifestyle, cur_data_str, first_name

    INPUT: 
    persona: The Persona class instance 
    wake_up_hour: an integer that indicates when the hour the persona wakes up 
                    (e.g., 8)
    OUTPUT: 
    a list of daily actions in broad strokes.

    EXAMPLE OUTPUT: 
    ['wake up and complete the morning routine at 6:00 am', 
        'have breakfast and brush teeth at 6:30 am',
        'work on painting project from 8:00 am to 12:00 pm', 
        'have lunch at 12:00 pm', 
        'take a break and watch TV from 2:00 pm to 4:00 pm', 
        'work on painting project from 4:00 pm to 6:00 pm', 
        'have dinner at 6:00 pm', 'watch TV from 7:00 pm to 8:00 pm']
    """

    def _create_prompt_input(self, **kwargs):
        test_input = kwargs.get("test_input")
        if test_input:
            return test_input

        persona = kwargs.get("persona")
        if not persona:
            raise ValueError("Persona is required")
        
        wake_up_hour = kwargs.get("wake_up_hour")
        if not wake_up_hour:
            raise ValueError("Wake up hour is required")
        
        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 am"]
        return prompt_input
    
    def _func_clean_up(self, **kwargs):
        gpt_response = kwargs.get("gpt_response")
        cr = []
        _cr = gpt_response.split(")")
        for i in _cr:
            if i[-1].isdigit():
                i = i[:-1].strip()
                if i[-1] == "." or i[-1] == ",":
                    cr += [i[:-1].strip()]
        return cr
    
    def _func_validate(self, **kwargs):
        gpt_response = kwargs.get("gpt_response")
        if not gpt_response:
            raise ValueError("GPT response is required for validation")
        
        try: self._func_clean_up(gpt_response=gpt_response)
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
        return True

    def _func_fail_safe(self):
        fs = ['wake up and complete the morning routine at 6:00 am', 
            'eat breakfast at 7:00 am', 
            'read a book from 8:00 am to 12:00 pm', 
            'have lunch at 12:00 pm', 
            'take a nap from 1:00 pm to 4:00 pm', 
            'relax and watch TV from 7:00 pm to 8:00 pm', 
            'go to bed at 11:00 pm'] 
        return fs
    
    def run(self, **kwargs):
        persona = kwargs.get("persona")
        if not persona:
            raise ValueError("Persona is required")
        
        test_input = kwargs.get("test_input")
        
        wake_up_hour = kwargs.get("wake_up_hour")
        if not wake_up_hour:
            raise ValueError("Wake up hour is required")
        
        gpt_param = {"engine": TEXT_DAVINCI_003, "max_tokens": 500, 
                    "temperature": 1, "top_p": 1, "stream": False,
                    "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
        
        prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
        prompt_input = self._create_prompt_input(persona=persona, wake_up_hour=wake_up_hour, test_input=test_input)
        prompt = RunGptPromptsBase.generate_prompt(prompt_input, prompt_template)
        fail_safe = self._func_fail_safe()

        output, raw_output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                        self._func_validate, self._func_clean_up)
        output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
                    + output)

        if debug or self.verbose: 
            print_run_prompts(prompt_template, persona, gpt_param, 
                            prompt_input, prompt, raw_output, output)

        return output, [output, prompt, gpt_param, prompt_input, fail_safe]