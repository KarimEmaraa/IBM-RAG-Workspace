from imports import *


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

query = "What are the fundamentals of quantum computing?"
query_vector = embeddings.embed_query(query)

print(query_vector[:10])

