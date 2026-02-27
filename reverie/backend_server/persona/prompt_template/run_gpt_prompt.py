"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: run_gpt_prompt.py
Description: Thin re-export shim. All prompt functions now live in the
run_gpt_prompts/ package (one function per file). This file re-exports
everything so that existing imports continue to work unchanged.
"""
from persona.prompt_template.run_gpt_prompts import *
from persona.prompt_template.run_gpt_prompts._helpers import (
    get_random_alphanumeric,
    extract_first_json_dict,
)
