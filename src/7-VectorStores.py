from imports import *
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

documents = [
    Document(
        page_content="Gemini Embedding 2 is Google's multimodal embedding architecture supporting long contexts up to 8k tokens.",
        metadata={"source": "gemini_guide.md", "category": "AI"}
    ),
    Document(
        page_content="LangChain provides vector store abstractions to simplify indexing, metadata filtering, and retriever creation.",
        metadata={"source": "langchain_guide.md", "category": "Framework"}
    ),
    Document(
        page_content="Photosynthesis converts solar photons into chemical energy stored in glucose bonds within chloroplasts.",
        metadata={"source": "biology_notes.md", "category": "Science"}
    )
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(documents)


vector_store = Chroma.from_documents(split_docs, embedding=embeddings)


print(f"First similarity search result: {vector_store.similarity_search('chemistry', k=2)[0].metadata['source']}")