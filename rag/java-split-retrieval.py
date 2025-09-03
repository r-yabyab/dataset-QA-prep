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
retriever = index.as_retriever(similarity_top_k=5)

# Step 5: Run interactive retrieval loop
while True:
    query = input("\nQuery: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    top_k = 2
    retrieved_nodes = retriever.retrieve(query)

    if not retrieved_nodes:
        print("No chunks retrieved for this query.")
        continue

    print(f"\nTop {top_k} retrieved chunks for query: '{query}'\n")
    for i, node in enumerate(retrieved_nodes, start=1):
        print(f"--- Node {i} ---")
        print(f"Source: {node.metadata.get('file_path', 'N/A')}")
        print(node.text)
        print("\n")