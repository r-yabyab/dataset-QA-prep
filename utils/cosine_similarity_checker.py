import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# === MODEL SETUP ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-unimodal")
model = AutoModel.from_pretrained("microsoft/unixcoder-base-unimodal")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()[0]

def load_jsonl_data(file_path):
    """Load data from JSONL file and extract Answer text for comparison (matching compare_removedupes.py)."""
    data = []
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                json_data = json.loads(line)
                # Extract only the Answer field for comparison (matching compare_removedupes.py)
                answer = json_data.get('Answer', '').strip()
                
                if answer:  # Only include non-empty answers
                    data.append({
                        'line_num': line_num,
                        'json_data': json_data,
                        'text': answer
                    })
                    texts.append(answer)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return data, texts

def compute_cosine_similarities(texts, threshold=0.9921):
    """Compute cosine similarities between all text pairs using code embeddings."""
    if len(texts) < 2:
        print("Need at least 2 texts to compare.")
        return []
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        embeddings.append(get_embedding(text))
    
    # Compute cosine similarity matrix
    print("Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find pairs with high similarity
    similar_pairs = []
    n = len(texts)
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i][j]
            if similarity >= threshold:
                similar_pairs.append({
                    'index1': i,
                    'index2': j,
                    'similarity': similarity
                })
    
    return similar_pairs

def compute_cosine_similarities_with_incremental_write(texts, data, threshold=0.97, rejected_jsonl=None):
    """Compute cosine similarities using code embeddings and write rejected lines incrementally every 10 matches."""
    if len(texts) < 2:
        print("Need at least 2 texts to compare.")
        return []
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        embeddings.append(get_embedding(text))
    
    # Compute cosine similarity matrix
    print("Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find pairs with high similarity and write incrementally
    similar_pairs = []
    rejected_indices = set()
    n = len(texts)
    
    # Open rejected file for incremental writing
    rejected_file = None
    if rejected_jsonl:
        rejected_file = open(rejected_jsonl, 'w', encoding='utf-8')
        print(f"Writing rejected duplicates incrementally to: {rejected_jsonl}")
    
    try:
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    similar_pairs.append({
                        'index1': i,
                        'index2': j,
                        'similarity': similarity
                    })
                    
                    # Identify the index to reject (higher index)
                    reject_idx = max(i, j)
                    if reject_idx not in rejected_indices:
                        rejected_indices.add(reject_idx)
                        
                        # Write to file immediately if we have a file open
                        if rejected_file:
                            line = json.dumps(data[reject_idx]['json_data'], ensure_ascii=False)
                            rejected_file.write(line + '\n')
                        
                        # Progress update every 10 rejections
                        if len(rejected_indices) % 10 == 0:
                            print(f"Progress: {len(rejected_indices)} duplicates identified and written...")
                            if rejected_file:
                                rejected_file.flush()  # Ensure data is written to disk
    
    finally:
        if rejected_file:
            rejected_file.close()
    
    print(f"Incremental writing complete. Total rejected: {len(rejected_indices)}")
    return similar_pairs

def analyze_jsonl_similarities(input_file, threshold=0.97, output_file=None, rejected_jsonl=None, clean_output_jsonl=None):
    """Analyze similarities in a JSONL file and optionally save results."""
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return
    
    print(f"Loading data from: {input_file}")
    data, texts = load_jsonl_data(input_file)
    
    print(f"Loaded {len(data)} valid entries")
    
    if len(data) < 2:
        print("Need at least 2 entries to compare similarities.")
        return
    
    # Compute similarities with incremental writing
    similar_pairs = compute_cosine_similarities_with_incremental_write(
        texts, data, threshold, rejected_jsonl
    )
    
    print(f"\nFound {len(similar_pairs)} pairs with similarity >= {threshold}")
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Count total rejected indices for reporting
    rejected_indices = set()
    for pair in similar_pairs:
        idx1, idx2 = pair['index1'], pair['index2']
        rejected_indices.add(max(idx1, idx2))
    
    print(f"Total {len(rejected_indices)} lines identified as duplicates")
    
    # Display results
    print(f"\nTop similar pairs:")
    print("-" * 80)
    
    for i, pair in enumerate(similar_pairs[:10]):  # Show top 10
        idx1, idx2 = pair['index1'], pair['index2']
        similarity = pair['similarity']
        
        print(f"\nPair {i+1}: Similarity = {similarity:.4f}")
        print(f"Line {data[idx1]['line_num']} vs Line {data[idx2]['line_num']}")
        print(f"Text 1 (first 100 chars): {data[idx1]['text'][:100]}...")
        print(f"Text 2 (first 100 chars): {data[idx2]['text'][:100]}...")
        print("-" * 40)
    
    # Save detailed results if output file specified
    if output_file:
        print(f"\nSaving detailed results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'input_file': input_file,
                'threshold': float(threshold),  # Convert to Python float
                'total_entries': len(data),
                'rejected_count': len(rejected_indices),
                'similar_pairs_count': len(similar_pairs),
                'similar_pairs': [
                    {
                        'line1': data[pair['index1']]['line_num'],
                        'line2': data[pair['index2']]['line_num'],
                        'similarity': float(pair['similarity']),  # Convert numpy float32 to Python float
                        'text1': data[pair['index1']]['text'],
                        'text2': data[pair['index2']]['text'],
                        'json1': data[pair['index1']]['json_data'],
                        'json2': data[pair['index2']]['json_data']
                    }
                    for pair in similar_pairs
                ]
            }, f, indent=2, ensure_ascii=False)
    
    # Write clean output file (input minus all duplicates)
    if clean_output_jsonl:
        print(f"\nCreating clean output file: {clean_output_jsonl}")
        with open(clean_output_jsonl, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(data):
                if i not in rejected_indices:  # Keep only non-duplicate entries
                    line = json.dumps(entry['json_data'], ensure_ascii=False)
                    f.write(line + '\n')
        
        clean_count = len(data) - len(rejected_indices)
        print(f"Clean output saved with {clean_count} entries (removed {len(rejected_indices)} duplicates)")
    
    return similar_pairs

if __name__ == "__main__":
    # Configuration
    input_file = r"W:\Users\cayab\dataset-QA-prep\data\java-data\parsed_java_functions_removesmall.jsonl"
    output_file = r"w:\Users\cayab\dataset-QA-prep\data\java-data\similarity_results.json"
    rejected_jsonl = r"w:\Users\cayab\dataset-QA-prep\data\java-data\rejected_duplicates.jsonl"
    clean_output_jsonl = r"w:\Users\cayab\dataset-QA-prep\data\java-data\parsed_java_functions_clean.jsonl"
    similarity_threshold = 0.97  # Adjust this threshold as needed (matching compare_removedupes.py)
    
    print(f"Analyzing similarities in: {input_file}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Results will be saved to: {output_file}")
    print(f"Rejected duplicates will be saved to: {rejected_jsonl}")
    print(f"Clean output (input minus duplicates) will be saved to: {clean_output_jsonl}")
    print("=" * 60)
    
    analyze_jsonl_similarities(input_file, similarity_threshold, output_file, rejected_jsonl, clean_output_jsonl)
