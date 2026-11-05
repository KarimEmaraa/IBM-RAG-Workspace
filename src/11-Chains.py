from imports import *
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch, RunnableLambda
parser = StrOutputParser()

idea_chain = (
    ChatPromptTemplate.from_template(
        "You are a helpful assistant. Generate a creative idea for a {topic}."
    )
    | QWEN_LLM
    | parser
)

critique_chain = (
    ChatPromptTemplate.from_template(
        "You are a helpful assistant. Critique the following idea: {idea}"
    )
    | QWEN_LLM
    | parser
)

full_chain = (
    RunnablePassthrough.assign(idea=idea_chain)
    | RunnableLambda(lambda x: print(f"Generated idea: {x['idea']}") or x)
    | RunnablePassthrough.assign(critique=critique_chain)
)

response = full_chain.invoke({"topic": "new mobile app"})
print(response)

pro_prompt = ChatPromptTemplate.from_template("List 2 short and direct pros of {technology}.")
con_prompt = ChatPromptTemplate.from_template("List 2 short and direct cons of {technology}.")

pro_chain = pro_prompt | QWEN_LLM | parser
con_chain = con_prompt | QWEN_LLM | parser

parallel_chain = RunnableParallel(
 pros=pro_chain,
 cons=con_chain
)

def word_counter(input: dict) -> str:
    pros = input.get("pros", "")
    cons = input.get("cons", "")
    total_words = len(pros.split()) + len(cons.split())
    return f"Total words in pros and cons: {total_words}"

all_chain = (
    parallel_chain
    | RunnableLambda(word_counter)
    | parser
)
result = all_chain.invoke(input="electric cars")
print(result)

classifier_chain = (
    ChatPromptTemplate.from_template(
        "Classify this user query into exactly one word: 'code', 'fitness', or 'general'.\nQuery: {query}"
    )
    | QWEN_LLM
    | parser
)

code_chain = ChatPromptTemplate.from_template("You are a Senior Python Engineer. Answer concisely: {query}") | QWEN_LLM | parser
fitness_chain = ChatPromptTemplate.from_template("You are an Athletic Coach. Answer energetically: {query}") | QWEN_LLM | parser
general_chain = ChatPromptTemplate.from_template("Answer helpfully: {query}") | QWEN_LLM | parser

branch = RunnableBranch(
    (lambda x: "code" in x["topic"].lower(), code_chain),
    (lambda x: "fitness" in x["topic"].lower(), fitness_chain),
    general_chain  # Default
)

router_chain = RunnablePassthrough.assign(topic=classifier_chain) | branch

print(router_chain.invoke({"query": "How do I reverse a list in Python?"}))





