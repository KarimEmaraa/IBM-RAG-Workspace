from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

pdf_loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762")
web_loader = WebBaseLoader("https://arxiv.org/abs/1706.03762")

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30, separator="\n")

pdf_documents = pdf_loader.load()
web_documents = web_loader.load()

print(f"Number of PDF documents: {len(pdf_documents)}")

chunks_pdf = splitter.split_documents(pdf_documents)
chunks_web = splitter.split_documents(web_documents)

# print(pdf_documents[0].page_content)
# print(web_documents[0].page_content)


print(f"Number of PDF documents: {len(chunks_pdf)}")