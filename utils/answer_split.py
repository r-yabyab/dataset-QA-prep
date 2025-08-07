import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from counttokens import count_tokens  # Your token counter

from pathlib import Path

# Entry point for repos
REPO_ROOT = Path("W:/Users/cayab/dataset-QA-prep/data/myrepos")
OUTPUT_ROOT = Path("W:/Users/cayab/dataset-QA-prep/data/outputs/answers")

rejected_files = []

REJECTED_OUTPUT_PATH = Path("W:/Users/cayab/dataset-QA-prep/data/outputs/rejected/rejected_files.txt")
REJECTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Supported extensions mapped to Langchain Language
EXT_TO_LANG = {
    ".js": Language.JS,
    ".ts": Language.TS,
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".cs": Language.CSHARP,
}

# Basic text fallback splitter
def plain_text_splitter(chunk_size=400, overlap=0):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

def get_splitter_for_file(file_path: Path):
    ext = file_path.suffix
    lang = EXT_TO_LANG.get(ext)
    if lang:
        separators = RecursiveCharacterTextSplitter.get_separators_for_language(lang)
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo",
            chunk_size=400,
            chunk_overlap=0,
            separators=separators,
        )
    elif ext == ".md":
        return plain_text_splitter()
    else:
        return None

def get_output_path(file_path: Path):
    relative = file_path.relative_to(REPO_ROOT)
    repo_name = relative.parts[0]
    out_path = OUTPUT_ROOT / repo_name / relative.with_suffix(".jsonl")
    os.makedirs(out_path.parent, exist_ok=True)
    return out_path

def process_file(file_path: Path):
    splitter = get_splitter_for_file(file_path)
    if not splitter:
        print(f"Skipping unsupported file: {file_path}")
        rejected_files.append(str(file_path))
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        rejected_files.append(str(file_path))
        return

    chunks = splitter.split_text(text)
    output_file = get_output_path(file_path)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for chunk in chunks:
            json_line = {"Question": "", "Answer": chunk}
            out_f.write(json.dumps(json_line) + "\n")

    print(f"Wrote {len(chunks)} chunks to {output_file}")

IGNORE_DIRS = {"node_modules", ".git", "__pycache__"}

def walk_repos():
    for repo in REPO_ROOT.iterdir():
        if repo.is_dir():
            for root, dirs, files in os.walk(repo):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]  # Ignore specific directories
                for file in files:
                    file_path = Path(root) / file
                    process_file(file_path)

if __name__ == "__main__":
    walk_repos()

    if rejected_files:
        with open(REJECTED_OUTPUT_PATH, "w", encoding="utf-8") as f:
            for line in rejected_files:
                f.write(line + "\n")
        print(f"Wrote {len(rejected_files)} rejected file paths to {REJECTED_OUTPUT_PATH}")