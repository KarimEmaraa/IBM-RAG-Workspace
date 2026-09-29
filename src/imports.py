from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from google import genai
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)