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
SIMILARITY_THRESHOLD = 0.996

# === MODEL SETUP ===
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/codebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModel.from_pretrained("sentence-transformers/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
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

# Only keep first 1/10th of the data
one_tenth_len = max(1, len(answers) // 10)
answers = answers[:one_tenth_len]
file_map = file_map[:one_tenth_len]

print(f"Using first 1/10th of answers: {len(answers)}")

# === EMBED CHUNKS ===
embeddings = []
for ans in tqdm(answers, desc="Embedding"):
    embeddings.append(get_embedding(ans))

# === COMPARE AND LOG ===
print("Computing similarities...")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

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

print(f"Logged {match_count} highly similar pairs to {LOG_PATH}")