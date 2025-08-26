from abc import ABC, abstractmethod

class RunGptPromptsBase(ABC):
    @staticmethod
    def generate_prompt(curr_input, prompt_lib_file): 
        """
        Takes in the current input (e.g. comment that you want to classifiy) and 
        the path to a prompt file. The prompt file contains the raw str prompt that
        will be used, which contains the following substr: !<INPUT>! -- this 
        function replaces this substr with the actual curr_input to produce the 
        final promopt that will be sent to the GPT3 server. 
        ARGS:
            curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                        INPUT, THIS CAN BE A LIST.)
            prompt_lib_file: the path to the promopt file. 
        RETURNS: 
            a str prompt that will be sent to OpenAI's GPT server.  
        """
        if type(curr_input) == type("string"): 
            curr_input = [curr_input]
        curr_input = [str(i) for i in curr_input]

        f = open(prompt_lib_file, "r")
        prompt = f.read()
        f.close()
        for count, i in enumerate(curr_input):   
            prompt = prompt.replace(f"!<INPUT {count}>!", i)
        if "<commentblockmarker>###</commentblockmarker>" in prompt: 
            prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
        return prompt.strip()

    @abstractmethod
    def _create_prompt_input(self, **kwargs):
        pass

    @abstractmethod
    def _func_clean_up(self, **kwargs):
        pass

    @abstractmethod
    def _func_validate(self, **kwargs):
        pass

    @abstractmethod
    def _func_fail_safe(self):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass