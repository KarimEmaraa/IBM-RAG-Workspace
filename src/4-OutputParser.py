from imports import *

from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")


person_info_query = "Tell me about einstein's age, name, and city."

person_json = JsonOutputParser(pydantic_object=Person)

person_comma = CommaSeparatedListOutputParser()

format_instructions_json = person_json.get_format_instructions()

format_instructions_comma = person_comma.get_format_instructions()

prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. {format_instructions} {person_info_query}"
)

chainJson = prompt_template | LLM | person_json

chainCommaSeparatedList = prompt_template | LLM | person_comma

#response = chainJson.invoke({"format_instructions": format_instructions_json, "person_info_query": person_info_query})

#print(response)  # Output: Person(name='Albert Einstein', age=76, city='Princeton')


response = chainCommaSeparatedList.invoke({"format_instructions": format_instructions_comma, "person_info_query": person_info_query})

print(response)  # Output: ['Albert Einstein', 76, 'Princeton']

