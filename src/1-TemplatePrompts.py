from imports import *

prompt_template = PromptTemplate.from_template("Explain to me about: {topic}.")

prompt = prompt_template.format(topic="programming")

client = genai.Client()


model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

chain = prompt_template | model | StrOutputParser()

response = chain.invoke({"topic": "programming"})
print(response)  # Output: A joke about programming