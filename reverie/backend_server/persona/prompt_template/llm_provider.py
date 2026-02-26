"""
File: llm_provider.py
Description: Abstraction layer for LLM backends. Switch between OpenAI and
Ollama by setting llm_provider in utils.py. All LLM calls in gpt_structure.py
go through this module.
"""
import sys
sys.path.append('../../')

from utils import *

if llm_provider == "ollama":
  from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings

  _chat_llm = ChatOllama(model=ollama_model, base_url=ollama_base_url)
  _embeddings = OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_base_url)

  def chat_request(prompt, model=None):
    response = _chat_llm.invoke(prompt)
    return response.content

  def completion_request(prompt, gpt_parameter):
    llm = OllamaLLM(
      model=ollama_model,
      base_url=ollama_base_url,
      temperature=gpt_parameter.get("temperature", 0),
      top_p=gpt_parameter.get("top_p", 1),
      stop=gpt_parameter.get("stop", None),
    )
    return llm.invoke(prompt)

  def get_embedding_vec(text):
    return _embeddings.embed_query(text)

elif llm_provider == "openai":
  from langchain_openai import ChatOpenAI, OpenAIEmbeddings

  _chat_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
  _embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

  def chat_request(prompt, model="gpt-3.5-turbo"):
    llm = _chat_llm if model == "gpt-3.5-turbo" else ChatOpenAI(model=model, api_key=openai_api_key)
    response = llm.invoke(prompt)
    return response.content

  def completion_request(prompt, gpt_parameter):
    llm = ChatOpenAI(
      model=gpt_parameter.get("engine", "gpt-3.5-turbo"),
      api_key=openai_api_key,
      temperature=gpt_parameter.get("temperature", 0),
      max_tokens=gpt_parameter.get("max_tokens", None),
      top_p=gpt_parameter.get("top_p", 1),
      frequency_penalty=gpt_parameter.get("frequency_penalty", 0),
      presence_penalty=gpt_parameter.get("presence_penalty", 0),
      stop=gpt_parameter.get("stop", None),
    )
    response = llm.invoke(prompt)
    return response.content

  def get_embedding_vec(text):
    return _embeddings.embed_query(text)

elif llm_provider == "claude-proxy":
  from langchain_openai import ChatOpenAI

  _chat_llm = ChatOpenAI(
    model="claude-sonnet-4-6",
    api_key="sk-cli-proxy-api",
    base_url="http://localhost:8317/v1",
  )

  def chat_request(prompt, model="claude-sonnet-4-6"):
    llm = _chat_llm if model == "claude-sonnet-4-6" else ChatOpenAI(
      model=model, api_key="sk-cli-proxy-api", base_url="http://localhost:8317/v1",
    )
    response = llm.invoke(prompt)
    return response.content

  def completion_request(prompt, gpt_parameter):
    # Filter out whitespace-only stop sequences (Claude rejects them)
    stop = gpt_parameter.get("stop", None)
    if stop:
      stop = [s for s in stop if s.strip()]
      if not stop:
        stop = None
    force_sonnet = True
    if force_sonnet:
      model = "claude-sonnet-4-6"
    else:
      model = gpt_parameter.get("engine", "claude-sonnet-4-6")
    llm = ChatOpenAI(
      model=model,
      api_key="sk-cli-proxy-api",
      base_url="http://localhost:8317/v1",
      temperature=gpt_parameter.get("temperature", 0),
      max_tokens=gpt_parameter.get("max_tokens", 4096),
      top_p=gpt_parameter.get("top_p", 1),
      stop=stop,
    )
    response = llm.invoke(prompt)
    return response.content

  from langchain_ollama import OllamaEmbeddings
  _embeddings = OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_base_url)

  def get_embedding_vec(text):
    return _embeddings.embed_query(text)

elif llm_provider == "vscode":
  from langchain_openai import ChatOpenAI

  _chat_llm = ChatOpenAI(
    model=vscode_model,
    api_key=vscode_api_key,
    base_url=vscode_base_url,
  )

  def chat_request(prompt, model=None):
    llm = _chat_llm if not model or model == vscode_model else ChatOpenAI(
      model=model, api_key=vscode_api_key, base_url=vscode_base_url,
    )
    response = llm.invoke(prompt)
    return response.content

  def completion_request(prompt, gpt_parameter):
    # Filter out whitespace-only stop sequences
    stop = gpt_parameter.get("stop", None)
    if stop:
      stop = [s for s in stop if s.strip()]
      if not stop:
        stop = None
    llm = ChatOpenAI(
      model=vscode_model,
      api_key=vscode_api_key,
      base_url=vscode_base_url,
      temperature=gpt_parameter.get("temperature", 0),
      max_tokens=gpt_parameter.get("max_tokens", 4096),
      top_p=gpt_parameter.get("top_p", 1),
      stop=stop,
    )
    response = llm.invoke(prompt)
    return response.content

  from langchain_ollama import OllamaEmbeddings
  _embeddings = OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_base_url)

  def get_embedding_vec(text):
    return _embeddings.embed_query(text)

else:
  raise ValueError(f"Unknown llm_provider: {llm_provider}. Use 'openai', 'ollama', 'claude-proxy', or 'vscode'.")
