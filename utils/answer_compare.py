from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Example
emb1 = get_embedding("function sum(a, b) { return a + b; }")
emb2 = get_embedding("const sum = (a, b) => a + b;")

similarity = cosine_similarity([emb1], [emb2])[0][0]
print("Similarity:", similarity)