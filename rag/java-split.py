from config import VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.core.node_parser import LangchainNodeParser
import chromadb

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Step 1: Load documents from Algorithms folder
documents = SimpleDirectoryReader(
    input_dir="../data/java-data-good/LeetCode-in-Java/Algorithms",
    recursive=True,  
    required_exts=[".java"],  # only pick up Java files
).load_data()
print(f"Total Java documents loaded: {len(documents)}")

# Step 2: Set up LangChain Java code splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA,
    chunk_size=512,
    chunk_overlap=50,
)

# Wrap it with LlamaIndex’s node parser
node_parser = LangchainNodeParser(splitter)

# Convert docs → nodes (chunks)
nodes = node_parser.get_nodes_from_documents(documents)
print(f"Total split nodes: {len(nodes)}")

# 🔹 Step 2.5: Log chunks to a text file
with open("chunks_log.txt", "w", encoding="utf-8") as f:
    for i, node in enumerate(nodes, start=1):
        f.write(f"=== Chunk {i} ===\n")
        f.write(f"Source: {node.metadata.get('file_path', 'N/A')}\n\n")
        f.write(node.text.strip())
        f.write("\n\n")

print("Chunks written to chunks_log.txt")

# Step 3: Load embedding model
embed_model = HuggingFaceEmbedding(model_name="microsoft/unixcoder-base")


# Step 4: Set up persistent ChromaDB vector store
db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
chroma_collection = db.get_or_create_collection(name=COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Step 5: Build and persist index
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("vectordb created")