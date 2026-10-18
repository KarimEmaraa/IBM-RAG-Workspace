from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document
from google import genai
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_ollama import ChatOllama


GOOGLE_LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.17)
QWEN_LLM = ChatOllama(model="qwen3:8b", temperature=0.7, reasoning=False, base_url="http://host.docker.internal:11434")