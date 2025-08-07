import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# === CONFIG ===
OUTPUT_ROOT = Path("W:/Users/cayab/dataset-QA-prep/data/outputs/answers")
LOG_PATH = Path("W:/Users/cayab/dataset-QA-prep/data/outputs/similar_chunks.txt")
SIMILARITY_THRESHOLD = 0.97

# === MODEL SETUP ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-unimodal")
model = AutoModel.from_pretrained("microsoft/unixcoder-base-unimodal")
model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model = model.to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()[0]

# === LOAD CHUNKS ===
answers = []
file_map = []  # (file_path, line_number)
print("Loading answers...")

for dirpath, _, filenames in os.walk(OUTPUT_ROOT):
    for fname in filenames:
        if fname.endswith(".jsonl"):
            fpath = Path(dirpath) / fname
            with open(fpath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        obj = json.loads(line)
                        answer = obj.get("Answer", "").strip()
                        if answer:
                            answers.append(answer)
                            ext = fpath.suffix  # e.g., ".js"
                            file_map.append((str(fpath), i + 1, ext))
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line in {fpath}")

print(f"Loaded {len(answers)} answers")

# # Only keep first 1/10th of the data
# one_tenth_len = max(1, len(answers) // 10)
# answers = answers[:one_tenth_len]
# file_map = file_map[:one_tenth_len]

# print(f"Using first 1/10th of answers: {len(answers)}")

# === EMBED CHUNKS ===
embeddings = []
for ans in tqdm(answers, desc="Embedding"):
    embeddings.append(get_embedding(ans))

# === COMPARE AND LOG ===
print("Computing similarities...")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Track which entries to remove: set of (file_path, line_number)
entries_to_remove = set()

with open(LOG_PATH, "w", encoding="utf-8") as log_file:
    match_count = 0
    for i in tqdm(range(len(embeddings)), desc="Comparing"):
        for j in range(i + 1, len(embeddings)):
            file1, line1, ext1 = file_map[i]
            file2, line2, ext2 = file_map[j]

            if ext1 != ext2:
                continue  # skip comparison if file extensions differ

            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= SIMILARITY_THRESHOLD:
                match_count += 1
                log_file.write(
                    f"\n[Similarity: {sim:.4f}]\n"
                    f"{file1}:{line1}\n"
                    f"{answers[i]}\n"
                    f"---- VS ----\n"
                    f"{file2}:{line2}\n"
                    f"{answers[j]}\n"
                    f"{'='*80}\n"
                )
                # Mark the second entry (j) for removal
                entries_to_remove.add((file2, line2))

print(f"Logged {match_count} highly similar pairs to {LOG_PATH}")

# === REMOVE DUPLICATES FROM FILES ===
if entries_to_remove:
    print(f"Removing {len(entries_to_remove)} duplicate entries from files...")
    
    # Group removals by file
    files_to_process = {}
    for file_path, line_num in entries_to_remove:
        if file_path not in files_to_process:
            files_to_process[file_path] = set()
        files_to_process[file_path].add(line_num)
    
    total_removed = 0
    # Process each file
    for file_path, lines_to_remove in files_to_process.items():
        print(f"Processing {file_path}: removing {len(lines_to_remove)} lines")
        
        # Read all lines from the file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Create new content excluding the lines to remove
        new_lines = []
        removed_count = 0
        for line_num, line in enumerate(lines, 1):
            if line_num not in lines_to_remove:
                new_lines.append(line)
            else:
                removed_count += 1
        
        # Write back the filtered content
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        
        total_removed += removed_count
    
    print(f"Duplicate removal completed! Total duplicates removed: {total_removed}")
else:
    print("No duplicates found to remove.")