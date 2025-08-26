from config import VECTOR_DB_PATH, COLLECTION_NAME
import chromadb
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel
import torch

CHECKPOINT_PATH = "./outputs-full/checkpoint-110"
MAX_SEQ_LENGTH = 512
DTYPE = None
LOAD_IN_4BIT = True

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load vector database
db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = db.get_or_create_collection(name=COLLECTION_NAME)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)

def get_rag_context(query: str, top_k: int = 3):
    """Get relevant context from RAG database"""
    query_embedding = embed_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return ""

def generate_response(prompt: str):
    """Generate response using your exact prompt"""
    # Get RAG context
    context = get_rag_context(prompt)
    
    # Use your prompt with context if available
    if context:
        full_prompt = f"{context}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(full_prompt):].strip()

if __name__ == "__main__":
    print("Ready! Type 'exit' to quit.")
    
    while True:
        prompt = input("\nEnter your prompt: ")
        
        if prompt.lower() in ['exit', 'quit']:
            break
            
        response = generate_response(prompt)
        print(f"\nResponse: {response}")
    
    print("Goodbye!")
