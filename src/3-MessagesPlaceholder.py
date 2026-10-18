from imports import *

message = ChatPromptTemplate.from_messages(
    [
        ("system", "You need to greet the user and ask them how they are doing."),
        MessagesPlaceholder("age"),
        ("human", "What is my age?"),
    ]
)

age_message = {
    "age": [
        HumanMessage(content="I am 20 years old."),
    ]
}

chain = message | LLM | StrOutputParser()

print(chain.invoke(input=age_message))


