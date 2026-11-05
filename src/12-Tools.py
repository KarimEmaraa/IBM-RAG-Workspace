from imports import *

messages = [ ("system", "You are a helpful assistant. You need first to greet the user. His name is karim."),
             ("human", "What is 5 + 8?") ]

@tool
def add(a: int, b: int) -> int:
    """
    A tool that adds two numbers together.
    """
    return a + b

llm_with_tools = QWEN_LLM.bind_tools([add])

aimessages = llm_with_tools.invoke(messages)
print(aimessages)


