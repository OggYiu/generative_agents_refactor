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
  import openai
  openai.api_key = openai_api_key

  def chat_request(prompt, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

  def completion_request(prompt, gpt_parameter):
    response = openai.Completion.create(
      model=gpt_parameter["engine"],
      prompt=prompt,
      temperature=gpt_parameter["temperature"],
      max_tokens=gpt_parameter["max_tokens"],
      top_p=gpt_parameter["top_p"],
      frequency_penalty=gpt_parameter["frequency_penalty"],
      presence_penalty=gpt_parameter["presence_penalty"],
      stream=gpt_parameter["stream"],
      stop=gpt_parameter["stop"],
    )
    return response.choices[0].text

  def get_embedding_vec(text):
    return openai.Embedding.create(
      input=[text], model="text-embedding-ada-002"
    )['data'][0]['embedding']

else:
  raise ValueError(f"Unknown llm_provider: {llm_provider}. Use 'openai' or 'ollama'.")
