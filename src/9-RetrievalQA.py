from imports import *
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

docs = PyPDFLoader("https://arxiv.org/pdf/2608.29401").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[0].page_content[:10]}")
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b", base_url="http://host.docker.internal:11434"
)
## Use custom prompt to reduce hallucinations and improve answer quality.
prompt_template= PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the following context to answer the question.
    \n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""
)

vector_store = Chroma.from_documents(chunks, embedding=embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=QWEN_LLM,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

results = qa_chain.invoke("List the authors of the paper.")

print(f"Answer: {results['result']}")