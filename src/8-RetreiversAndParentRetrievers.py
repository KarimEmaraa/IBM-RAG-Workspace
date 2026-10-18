from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever

from imports import *

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-1-preview",
    output_dimensionality=768
)

docs = [
    Document(
        page_content="Gemini Embedding 2 is Google's multimodal embedding architecture supporting long contexts up to 8k tokens.",
        metadata={"source": "gemini_guide.md", "category": "AI"}
    ),
    Document(
        page_content="LangChain provides vector store abstractions to simplify indexing, metadata filtering, and retriever creation.",
        metadata={"source": "langchain_guide.md", "category": "Framework"}
    ),
    Document(page_content="Pharoes were the rulers of ancient Egypt, often considered divine and responsible for maintaining ma'at, or cosmic order.",
        metadata={"source": "history_notes.md", "category": "History"}
    )
]   

vectore_store = Chroma.from_documents(docs, embedding=embeddings)

vector_retriever = vectore_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

vector_retreiver2 = vectore_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

results = vector_retriever.invoke("What is the role of pharaohs in ancient Egypt?")

print(f"First similarity search result: {results[0].metadata['source']}")

results2 = vector_retreiver2.invoke("What is the role of pharaohs in ancient Egypt?")

print(f"First similarity score threshold search result: {results2[0].metadata['source']}")



## Parent retriever example

long_docs = Document(page_content="""
    LangChain was launched in October 2022 as an open source project by Harrison Chase, while working at machine learning startup Robust Intelligence. In April 2023, LangChain had incorporated and the new startup raised over $20 million in funding at a valuation of at least $200 million from venture firm Sequoia Capital, a week after announcing a $10 million seed investment from Benchmark.[3][4]

In the third quarter of 2023, the LangChain Expression Language (LCEL) was introduced, which provides a declarative way to define chains of actions.[5][6]

In October 2023 LangChain introduced LangServe, a deployment tool to host LCEL code as a production-ready API.[7]

In January 2024, LangChain introduced LangGraph, an open-source framework designed to facilitate the development of stateful and multi-agent applications using LLMs by representing the workflows of agents using graphs, thereby resolving issues faced while developing workflows in the LangChain Expression Language.

In February 2024 LangChain released LangSmith, a closed-source observability and evaluation platform for LLM applications, and announced a US $25 million Series A led by Sequoia Capital.[8] On 14 May 2025 the company launched LangGraph Platform into general availability, providing managed infrastructure for deploying long-running, stateful AI agents.[9]

In April 2025, LangChain was featured in the Forbes AI 50 list.[10]

In October 2025, LangChain made an investment of $125 million via Series B under the leadership of IVP at the valuation of $1.25 billion. The participants in this financing include Sequoia Capital and Benchmark.
""")

child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators="\n")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0, separators="\n")

vector_store = Chroma(embedding_function=embeddings)

doc_store = InMemoryStore()

retreiver = ParentDocumentRetriever(
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
    vectorstore=vector_store,
    docstore=doc_store)


retreiver.add_documents([long_docs])


retrieved_docs = retreiver.invoke("When is LangSmith released?")

for r in retrieved_docs:
    print(f"Retrieved document: {r.page_content}\n --- \n") 


