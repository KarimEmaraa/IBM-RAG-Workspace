import os
import chromadb
from imports import *
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# Define your storage directory
DB_PATH = "./chroma_db"
COLLECTION_NAME = "qwen_simulated_docs"

# 1. Initialize the embedding model via Ollama
embeddings = OllamaEmbedding(
    model_name="qwen3-embedding:8b",
    base_url="http://host.docker.internal:11434",
)

# 2. Initialize Chroma PersistentClient pointing to a local directory
chroma_client = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# 3. Connect Chroma to LlamaIndex vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Check if the collection already contains data
if chroma_collection.count() == 0:
    print("Collection is empty. Indexing simulated documents...")
    
    simulated_data = [
        # --- AI & Machine Learning ---
        {
            "text": "Deep learning architectures have revolutionized natural language processing tasks.",
            "meta": {"source": "ml_research_journal", "year": 2024, "category": "AI", "doc_id": "doc_001"},
        },
        {
            "text": "Low-Rank Adaptation (LoRA) significantly cuts down fine-tuning memory by injecting trainable low-rank decomposition matrices into attention weights.",
            "meta": {"source": "arxiv_llm_papers", "year": 2024, "category": "AI", "doc_id": "doc_005"},
        },
        {
            "text": "Quantization techniques like AWQ and GPTQ enable running 70B parameter models on consumer-grade hardware with negligible perplexity degradation.",
            "meta": {"source": "hardware_acceleration_review", "year": 2025, "category": "AI", "doc_id": "doc_006"},
        },
        {
            "text": "Mixture of Experts (MoE) dynamically activates sparse sub-networks during inference, allowing higher model capacity at lower compute overhead.",
            "meta": {"source": "ai_systems_conf", "year": 2025, "category": "AI", "doc_id": "doc_007"},
        },

        # --- Vector Databases & Search ---
        {
            "text": "Vector databases enable high-dimensional approximate nearest neighbor searches for RAG systems.",
            "meta": {"source": "data_engineering_notes", "year": 2025, "category": "Database", "doc_id": "doc_003"},
        },
        {
            "text": "Hierarchical Navigable Small World (HNSW) graphs offer logarithmic search scaling and high recall for multi-million vector datasets.",
            "meta": {"source": "vector_benchmarks", "year": 2024, "category": "Database", "doc_id": "doc_008"},
        },
        {
            "text": "Hybrid search architectures pair dense neural embeddings with BM25 lexical sparse vectors to capture exact keyword and semantic matches simultaneously.",
            "meta": {"source": "information_retrieval_quarterly", "year": 2025, "category": "Database", "doc_id": "doc_009"},
        },
        {
            "text": "Cross-encoder rerankers dramatically boost top-k retrieval relevance when placed downstream of fast bi-encoder vector lookups.",
            "meta": {"source": "rag_architecture_guide", "year": 2024, "category": "Database", "doc_id": "doc_010"},
        },

        # --- Infrastructure, Cloud & DevOps ---
        {
            "text": "Containerization with Docker simplifies multi-stage deployments across environments.",
            "meta": {"source": "devops_weekly", "year": 2023, "category": "Infrastructure", "doc_id": "doc_002"},
        },
        {
            "text": "Kubernetes Horizontal Pod Autoscalers (HPA) adapt workload replicas based on real-time Prometheus custom metrics and memory pressure.",
            "meta": {"source": "cloud_native_today", "year": 2024, "category": "Infrastructure", "doc_id": "doc_011"},
        },
        {
            "text": "Terraform state management with S3 remote backends and DynamoDB distributed locks prevents deployment race conditions across engineering teams.",
            "meta": {"source": "iac_patterns", "year": 2023, "category": "Infrastructure", "doc_id": "doc_012"},
        },

        # --- Backend & Python Development ---
        {
            "text": "Async programming in Python reduces IO-bound bottlenecking in microservices.",
            "meta": {"source": "python_best_practices", "year": 2024, "category": "Backend", "doc_id": "doc_004"},
        },
        {
            "text": "FastAPI leverages Pydantic models for fast request deserialization, automatic OpenAPI schema generation, and strict compile-time validation.",
            "meta": {"source": "backend_engineering_journal", "year": 2024, "category": "Backend", "doc_id": "doc_013"},
        },
        {
            "text": "Redis Pub/Sub combined with WebSockets enables high-throughput bidirectional event distribution across distributed web clients.",
            "meta": {"source": "distributed_systems_review", "year": 2025, "category": "Backend", "doc_id": "doc_014"},
        },

        # --- Security & Governance ---
        {
            "text": "Implementing Zero Trust architecture necessitates continuous cryptographic identity verification, least-privilege scoping, and strict microsegmentation.",
            "meta": {"source": "infosec_handbook", "year": 2024, "category": "Security", "doc_id": "doc_015"},
        },
        {
            "text": "Prompt injection defenses require token classification, input sandboxing, and separating system directives from untrusted user content.",
            "meta": {"source": "ai_safety_symposium", "year": 2025, "category": "Security", "doc_id": "doc_016"},
        },
    ]

    documents = [
        Document(
            text=item["text"],
            metadata=item["meta"],
            doc_id=str(id),
        )
        for id, item in enumerate(simulated_data)
    ]

    # Embeds and writes documents directly to disk via ChromaDB
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embeddings,
    )
else:
    print(f"Loading existing collection with {chroma_collection.count()} documents...")
    
    # Reload existing index directly from Chroma without re-embedding
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embeddings,
    )

# 5. Query the persistent index
retriever = index.as_retriever(similarity_top_k=2)
results = retriever.retrieve("How do vector databases handle similarity?")

for idx, node in enumerate(results, start=1):
    print(f"\nResult {idx}:")
    print(f"Text: {node.text}")
    print(f"Metadata: {node.metadata}")