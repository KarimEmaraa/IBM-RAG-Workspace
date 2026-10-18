from imports import *
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

history = InMemoryChatMessageHistory()

def chat_with_history(user_input):
    # Add user input to chat history
    history.add_user_message(user_input)

    reply = QWEN_LLM.invoke(history.messages)

    history.add_ai_message(reply.content)

    return reply.content


# print(chat_with_history("My name is Karim and I love hiking."))
# print(chat_with_history("What is my name and hobby?"))


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{user_input}"),
    ]
)

chain = prompt | QWEN_LLM

store = {}

def get_session_history(session_id : str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)



cfg_alice = {"configurable": {"session_id": "alice"}}
cfg_bob   = {"configurable": {"session_id": "bob"}}

chat_chain.invoke({"user_input": "I'm Alice, I study biology."}, config=cfg_alice)
chat_chain.invoke({"user_input": "I'm Bob, I'm a chef."},        config=cfg_bob)

print(chat_chain.invoke({"user_input": "What do I do?"}, config=cfg_alice).content)  # biology
print(chat_chain.invoke({"user_input": "What do I do?"}, config=cfg_bob).content)

print(store["alice"].messages)