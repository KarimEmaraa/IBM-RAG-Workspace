from imports import *

prompt_template = PromptTemplate.from_template("Explain to me about: {topic}.")

prompt = prompt_template.format(topic="programming")

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

chain = prompt_template | model | StrOutputParser()

response = chain.invoke({"topic": "programming"})
print(response)


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