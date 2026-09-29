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

### Prompt templates

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You need first to greet the user. His name is {name}."),
    ("human", "{question}"),
    ("ai", "{answer}")
    ])


messages = template.format_messages(name="karim", question="What is 5 + 8?", answer="13 honey it is easy!")

print(messages)

### Few-shot prompting

examples = [
    {"name": "karim", "question": "What is 5 + 8?", "answer": "13 honey it is easy!" }, 
    {"name": "John", "question": "What is 10 - 3?", "answer": "7, that's a easy one!"}
]

example_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You need first to greet the user. His name is {name}."),
    ("human", "{question}"),
    ("ai", "{answer}")
    ])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You need first to greet the user. His name is {name}."),
    few_shot_prompt,
    ("human", "{question}")])

# Uncomment to send the message to llm.
#print(llm.invoke(final_prompt.format_messages(name="karim", question="What is 20 + 3?")).content[0]["text"])