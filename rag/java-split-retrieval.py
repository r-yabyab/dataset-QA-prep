from config import VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Step 1: Load embedding model
embed_model = HuggingFaceEmbedding(model_name="microsoft/unixcoder-base")
Settings.embed_model = embed_model

# Step 2: Connect to existing Chroma collection
db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
chroma_collection = db.get_or_create_collection(name=COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Step 3: Wrap vector store in storage context and create index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Step 4: Create retriever
retriever = index.as_retriever(similarity_top_k=4)

# Step 5: Run interactive retrieval loop
from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

while True:
    query = input("\nQuery: ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    # Retrieve top 4 candidates
    retriever.similarity_top_k = 4
    retrieved_nodes = retriever.retrieve(query)

    if not retrieved_nodes:
        print("No chunks retrieved.")
        continue

    # Rerank top 4
    pairs = [(query, node.text) for node in retrieved_nodes]
    scores = reranker.predict(pairs)

    # Get indices of top 2 scores (descending)
    top_indices = np.argsort(scores)[-2:][::-1]
    top_nodes = [retrieved_nodes[i] for i in top_indices]

    # Display top 2 reranked chunks
    for rank, node in enumerate(top_nodes, start=1):
        print(f"\n=== Top {rank} reranked chunk ===")
        print(f"Source: {node.metadata.get('file_path', 'N/A')}")
        print(node.text)