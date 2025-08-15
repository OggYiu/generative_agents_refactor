from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
import re

GENERAL_SYSTEM_PROMPT = """
"""

def remove_think_tags(text):
    """
    Removes <think> tags and their content from the given text.
    
    Args:
    text (str): The input text containing <think> tags.
    
    Returns:
    str: The text with <think> tags and content removed.
    """
    # Use regex to find and remove <think>.*?</think> including content
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip any leading/trailing whitespace that might result from removal
    return cleaned_text.strip()

class LLMModel:
    def __init__(self):
        pass

    def _create_llm(self, **kwargs):
        model_name = kwargs.get("model", "")
        if not model_name:
            raise ValueError("Model name cannot be empty")
        
        temperature = kwargs.get("temperature", 0.0)
        
        if "gpt" in model_name and not "oss" in model_name:
            llm = ChatOpenAI(model=model_name, temperature=temperature)
        else:
            llm = ChatOllama(model=model_name, temperature=temperature)
        return llm
    
    def create(self, **kwargs):
        llm = self._create_llm(**kwargs)
        schema = kwargs.get("schema", None)
        if schema is not None:
            llm = llm.with_structured_output(schema, include_raw=False)

        input = kwargs.get("input", "")
        if input is None:
            raise ValueError("Input cannot be None")
        
        messages = [
            ("system", GENERAL_SYSTEM_PROMPT),
            ("human", input),
        ]
        response = llm.invoke(messages)
        return response.content

class EmbeddingModel:
    def __init__(self):
        self.model_name = ""
        
    def create(self, **kwargs):
        model = kwargs.get("model", self.model_name)
        if model != self.model_name:
            self.embeddings = OllamaEmbeddings(model=model)
        input = kwargs.get("input", [])
        return self.embeddings.embed_documents(input)

