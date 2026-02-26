"""Quick test script for the Claude proxy via langchain ChatOpenAI."""
from langchain_openai import ChatOpenAI

model = "claude-sonnet-4-6"
gpt_parameter = {"temperature": 0, "max_tokens": 100, "top_p": 1}

llm = ChatOpenAI(
    model=model,
    api_key="sk-cli-proxy-api",
    base_url="http://localhost:8317/v1",
    temperature=gpt_parameter.get("temperature", 0),
    max_tokens=gpt_parameter.get("max_tokens", 4096),
    top_p=gpt_parameter.get("top_p", 1),
    stop=gpt_parameter.get("stop", None),
)

print("Sending test prompt...")
response = llm.invoke("Say hello in one sentence.")
print(f"Response: {response.content.encode('utf-8').decode('utf-8')}", flush=True)
