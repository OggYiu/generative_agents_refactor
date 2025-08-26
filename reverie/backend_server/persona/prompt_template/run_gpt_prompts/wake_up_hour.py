from persona.prompt_template.run_gpt_prompts.run_gpt_prompts_base import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *

from utils import *

class WakeUpHour(RunGptPromptsBase):
    """
    Given the persona, returns an integer that indicates the hour when the 
    persona wakes up.  

    INPUT: 
    persona: The Persona class instance 
    OUTPUT: 
    integer for the wake up hour.
    """

    def _create_prompt_input(self, **kwargs):
        test_input = kwargs.get("test_input")
        if test_input: return test_input

        persona = kwargs.get("persona")
        if not persona:
            raise ValueError("Persona is required")
        
        prompt_input = [persona.scratch.get_str_iss(),
                        persona.scratch.get_str_lifestyle(),
                        persona.scratch.get_str_firstname()]
        return prompt_input
    
    def _func_clean_up(self, **kwargs):
        gpt_response = kwargs.get("gpt_response")
        if not gpt_response:
            raise ValueError("GPT response is required for cleanup")

        cr = int(gpt_response.strip().lower().split("am")[0])
        return cr
    
    def _func_validate(self, **kwargs):
        gpt_response = kwargs.get("gpt_response")
        prompt = kwargs.get("prompt", "")
        try: self._func_clean_up(gpt_response=gpt_response)
        except: return False
        return True

    def _func_fail_safe(self): 
        fs = 8
        return fs
    
    def run(self, **kwargs):
        persona = kwargs.get("persona")
        if not persona:
            raise ValueError("Persona is required")
        
        test_input = kwargs.get("test_input")
        
        gpt_param = {
            "engine": TEXT_DAVINCI_002, 
            "max_tokens": 5, 
            "temperature": 0.8, 
            "top_p": 1, 
            "stream": False,
            "frequency_penalty": 0, 
            "presence_penalty": 0, 
            "stop": ["\n"]}
        prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
        prompt_input = self._create_prompt_input(persona=persona, test_input=test_input)
        prompt = RunGptPromptsBase.generate_prompt(prompt_input, prompt_template)
        
        schema = None
        # class WakeUpHourSchema(BaseModel):
        #   content: str = Field(description="the wake up hour of the person")
        # schema = WakeUpHourSchema

        verbose = False

        output, raw_output = safe_generate_response(
            prompt, 
            gpt_param, 
            5, 
            self._func_fail_safe,
            self._func_validate, 
            self._func_clean_up, 
            schema,
            verbose)
        
        if debug or verbose: 
            print_run_prompts(
                prompt_template,
                persona,
                gpt_param,
                prompt_input,
                prompt,
                raw_output,
                output)
            
        # return output, [output, prompt, gpt_param, prompt_input, self._func_fail_safe()]
        return output