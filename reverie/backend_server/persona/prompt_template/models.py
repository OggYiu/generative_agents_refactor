from langchain_ollama import OllamaEmbeddings

class LLM:
    def __init__(self):
        self.model_name = ""

class EmbeddingModel:
    def __init__(self):
        self.model_name = ""
        
    def create(self, **kwargs):
        model = kwargs.get("model", self.model_name)
        if model != self.model_name:
            self.embeddings = OllamaEmbeddings(model=model)
        input = kwargs.get("input", [])
        return self.embeddings.embed_documents(input)

