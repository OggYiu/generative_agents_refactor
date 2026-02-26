"""Quick test script for ChatOpenAI with base_url http://127.0.0.1:3030"""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key="sk-cli-proxy-api",
    base_url="http://127.0.0.1:3030/v1",
    temperature=0,
    max_tokens=100,
)

print("Sending test prompt to http://127.0.0.1:3030 ...")
response = llm.invoke("Say hello in one sentence.")
print(f"Response: {response.content}")
