from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from counttokens import count_tokens
import tiktoken
import os
import json

src_dir = r"W:/Users/cayab/dataset-QA-prep/data/swapples/src"
output_file = r"W:/Users/cayab/dataset-QA-prep/data/swapples/src_chunks.jsonl"

# js_splitter = RecursiveCharacterTextSplitter.from_language(

LANGUAGE_SEPARATORS = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
print(LANGUAGE_SEPARATORS)

print(tiktoken.list_encoding_names())

js_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-3.5-turbo",
    # chunk_size=600, # a bit too big, llama 552, gpt2 743
    # chunk_size=400,
    chunk_size=300,
    chunk_overlap=0,
    separators=LANGUAGE_SEPARATORS,
)

input_file = "../data/swapples/src/components/game/GameBoardComponent.js"

parts = input_file.replace("\\", "/").split("/")
data_index = parts.index("data")
project_name = parts[data_index + 1]
filename = os.path.basename(input_file)
base_filename = os.path.splitext(filename)[0]  
output_file_name = f"../data/outputs/{project_name}/{base_filename}.jsonl"

os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

    chunks = js_splitter.split_text(text)
    print(len(chunks))
    print(chunks[0])
    token_count = count_tokens(chunks[0])
    print(f"Token count for chunk 2: {token_count}")

    with open(output_file_name, "w", encoding="utf-8") as out_f:
        for chunk in chunks:
            json_line = {"Question": "", "Answer": chunk}
            out_f.write(json.dumps(json_line) + "\n")

# all_docs = []

# # Walk the src directory
# for root, dirs, files in os.walk(src_dir):
#     for file in files:
#         if file.endswith(".js"):
#             file_path = os.path.join(root, file)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 JS_CODE = f.read()

#             # Split the file's code into documents (each is a chunk)
#             docs = js_splitter.create_documents([JS_CODE])

#             # docs is a list of Document objects, get the page_content from each
#             for i, doc in enumerate(docs, 1):
#                 all_docs.append({
#                     "answer": f"{file}_chunk_{i:03d}",
#                     "content": doc.page_content
#                 })

# print(f"Total chunks created: {len(all_docs)}")

# # Write to JSONL file
# with open(output_file, "w", encoding="utf-8") as f:
#     for entry in all_docs:
#         f.write(json.dumps(entry) + "\n")

# print(f"Saved chunks to {output_file}")