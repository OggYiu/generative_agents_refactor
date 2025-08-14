from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

class LLMModel:
    def __init__(self):
        self.model_name = ""
        self.llm = None

    def create(self, **kwargs):
        model_name = kwargs.get("model", self.model_name)
        print(f"Creating LLM with model: {model_name}")
        # temperature = kwargs.get("temperature", 0.8)
        if model_name != self.model_name:
            # self.llm = ChatOllama(
            #     model = model,
            #     temperature = temperature,
            # )
            if "gpt" in model_name:
                self.llm = ChatOpenAI(model=model_name)
            else:
                self.llm = ChatOllama(model=model_name)
        self.model_name = model_name
        input = kwargs.get("input", "")
        if input is None:
            raise ValueError("Input cannot be None")
        messages = [
            ("system", ""),
            ("human", input),
        ]
        response = self.llm.invoke(messages)
        # print(f"LLM input: {input}")
        # print(f"LLM response: {response}")
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

