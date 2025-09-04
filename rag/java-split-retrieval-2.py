from config import VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import CrossEncoder
import chromadb
import re

# Step 1: Load embedding model and reranker
embed_model = HuggingFaceEmbedding(model_name="microsoft/unixcoder-base")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Step 2: Connect to existing ChromaDB vector store
db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
chroma_collection = db.get_collection(name=COLLECTION_NAME)  # get existing collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Step 3: Load existing index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)
print("Loaded existing vectordb for retrieval")

# Step 4: Query handling with reranking (get top 1st result)
def retrieve_with_prompt(prompt: str):
    # Split prompt into retrieval_prompt and rest_of_prompt
    # Extract first sentence more robustly
    first_sentence_match = re.match(r"^(.*?[\.!?])\s*(.*)", prompt, re.DOTALL)
    if first_sentence_match:
        retrieval_prompt = first_sentence_match.group(1).strip()
        rest_of_prompt = first_sentence_match.group(2).strip()
    else:
        # fallback: if no punctuation found, split at first space after 20 chars
        words = prompt.split()
        if len(words) > 3:
            retrieval_prompt = " ".join(words[:3])  # Use first 3 words for retrieval
            rest_of_prompt = " ".join(words[3:])
        else:
            retrieval_prompt = prompt
            rest_of_prompt = ""

    print(f"Retrieval prompt: {retrieval_prompt}")
    print(f"Rest of prompt: {rest_of_prompt[:100]}...")

    # Create a retriever from the index
    retriever = index.as_retriever(similarity_top_k=5)
    
    # Retrieve based on retrieval_prompt
    retrieved_nodes = retriever.retrieve(retrieval_prompt)
    
    # Rerank the retrieved nodes
    passages = [node.text for node in retrieved_nodes]
    scores = reranker.predict([(retrieval_prompt, passage) for passage in passages])
    
    # Get the index of the highest scoring passage
    best_idx = scores.argmax()
    best_passage = passages[best_idx]
    
    # Combine: retrieved context + rest of prompt
    if rest_of_prompt:
        combined_prompt = f"{best_passage}\n\n{rest_of_prompt}"
    else:
        combined_prompt = best_passage
    return combined_prompt

# Example usage
prompt = """
Check if string isPalindrome. Write a complete Java class named StringUtils that: 1. Has a public class declaration: "public class StringUtils {" 2. Has a public instance method "isPalindrome(String input)" that checks if input string is a palindrome. The class should work with JUnit tests where an instance is created directly in the test: "StringUtils utils = new StringUtils();". Example structure: "public class StringUtils { // your methods here }"
"""
# Check if string isPalindrome. Write a complete Java class named StringUtils that: 1. Has a public class declaration: "public class StringUtils {" 2. Has a public instance method "isPalindrome(String input)" that checks if input string is a palindrome. The class should work with JUnit tests where an instance is created directly in the test: "StringUtils utils = new StringUtils();". Example structure: "public class StringUtils { // your methods here }"

final_prompt = retrieve_with_prompt(prompt)
print(final_prompt)
