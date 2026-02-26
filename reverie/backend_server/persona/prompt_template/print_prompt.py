"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: print_prompt.py
Description: For printing prompts when the setting for verbose is set to True.
Also writes each GNS function call to a log file under logs/{timestamp}/.
"""
import sys
sys.path.append('../')

import inspect
import json
import numpy
import datetime
import os
import random

from global_methods import *
from persona.prompt_template.gpt_structure import *
from utils import *

##############################################################################
#                    PERSONA Chapter 1: Prompt Structures                    #
##############################################################################

# Create a session log directory with timestamp (once per run)
_log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "..", "logs", _log_timestamp)
os.makedirs(_log_dir, exist_ok=True)

# Track call counts per function name to avoid overwriting
_log_call_counts = {}


def _get_gns_function_name():
  """Derive the GNS function name from the caller's call stack.
  Looks for run_gpt_prompt_* in the stack and extracts the suffix,
  e.g. run_gpt_prompt_wake_up_hour -> generate_wake_up_hour.
  Falls back to the caller's caller function name."""
  for frame_info in inspect.stack():
    fname = frame_info.function
    if fname.startswith("run_gpt_prompt_"):
      return "generate_" + fname[len("run_gpt_prompt_"):]
  return "unknown"


def print_run_prompts(prompt_template=None,
                      persona=None,
                      gpt_param=None,
                      prompt_input=None,
                      prompt=None,
                      output=None):
  gns_function = _get_gns_function_name()

  # Build LLM settings summary
  llm_settings = {"llm_provider": llm_provider}
  if llm_provider == "vscode":
    llm_settings.update({"model": vscode_model, "base_url": vscode_base_url})
  elif llm_provider == "claude-proxy":
    llm_settings.update({"base_url": "http://localhost:8317/v1"})
  elif llm_provider == "ollama":
    llm_settings.update({"model": ollama_model, "base_url": ollama_base_url,
                         "embedding_model": ollama_embedding_model})
  elif llm_provider == "openai":
    llm_settings.update({"model": "gpt-3.5-turbo"})

  log_lines = []
  log_lines.append(f"GNS FUNCTION: <{gns_function}>")
  log_lines.append(f"=== {prompt_template}")
  log_lines.append("~~~ llm_settings -------------------------------------------------")
  log_lines.append(f"{llm_settings}\n")
  log_lines.append("~~~ persona    ---------------------------------------------------")
  log_lines.append(f"{persona.name}\n")
  log_lines.append("~~~ gpt_param ----------------------------------------------------")
  log_lines.append(f"{gpt_param}\n")
  log_lines.append("~~~ prompt_input    ----------------------------------------------")
  log_lines.append(f"{prompt_input}\n")
  log_lines.append("~~~ prompt    ----------------------------------------------------")
  log_lines.append(f"{prompt}\n")
  log_lines.append("~~~ output    ----------------------------------------------------")
  log_lines.append(f"{output}\n")
  log_lines.append("=== END ==========================================================")
  log_lines.append("\n\n\n")

  full_log = "\n".join(log_lines)

  # Print to console
  print(full_log)

  # Write to log file
  file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  filename = f"{file_timestamp}_{gns_function}.txt"
  log_path = os.path.join(_log_dir, filename)
  with open(log_path, "w", encoding="utf-8") as f:
    f.write(full_log)
