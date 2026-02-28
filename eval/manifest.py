"""
Eval manifest — maps each eval pair (prompt + golden file) to its
corresponding prompt module and metadata.

Each entry tells the test harness:
  - which module contains the clean_up / validate / fail_safe functions
  - which files hold the rendered prompt and expected golden output
  - what Python type the golden output should be compared as
"""

EVAL_CASES = {
    "generate_wake_up_hour": {
        "module": "persona.prompt_template.run_gpt_prompts.wake_up_hour",
        "prompt_file": "generate_wake_up_hour_prompt.txt",
        "golden_file": "generate_wake_up_hour_golden.txt",
        "golden_type": "int",
        "gpt_param": {
            "engine": "text-davinci-002",
            "max_tokens": 5,
            "temperature": 0.8,
            "top_p": 1,
            "stream": False,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["\n"],
        },
    },
    "generate_hourly_schedule_01": {
        "module": "persona.prompt_template.run_gpt_prompts.generate_hourly_schedule",
        "prompt_file": "generate_hourly_schedule_01_prompt.txt",
        "golden_file": "generate_hourly_schedule_01_golden.txt",
        "golden_type": "str",
        "gpt_param": {
            "engine": "text-davinci-003",
            "max_tokens": 50,
            "temperature": 0.5,
            "top_p": 1,
            "stream": False,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["\n"],
        },
    },
    "generate_first_daily_plan": {
        "module": "persona.prompt_template.run_gpt_prompts.daily_plan",
        "prompt_file": "generate_first_daily_plan_prompt.txt",
        "golden_file": "generate_first_daily_plan_golden.txt",
        "golden_type": "list",
        "gpt_param": {
            "engine": "text-davinci-003",
            "max_tokens": 500,
            "temperature": 1,
            "top_p": 1,
            "stream": False,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        },
    },
}
