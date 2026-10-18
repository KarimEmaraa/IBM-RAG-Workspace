from imports import *

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

# Create a list of messages to simulate a conversation
messages = [
    SystemMessage(content="You are a helpful assistant. You need first to greet the user. His name is karim."),
    HumanMessage(content="What is 5 + 8?"),
]

## or 

messages = [ ("system", "You are a helpful assistant. You need first to greet the user. His name is karim."),
             ("human", "What is 5 + 8?") ]

@tool
def add(a: int, b: int) -> int:
    """
    A tool that adds two numbers together.
    """
    return a + b

llm_with_tools = llm.bind_tools([add])

aimessages = llm_with_tools.invoke(messages)

messages.append(aimessages)

for call in aimessages.tool_calls:
    result = add.invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

# Uncomment to send the message to llm.
# print(llm_with_tools.invoke(messages).content[0]["text"])
