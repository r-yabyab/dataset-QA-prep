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
    # language=Language.JS,
    model_name="gpt-3.5-turbo",
    chunk_size=512,
    chunk_overlap=0,
    separators=LANGUAGE_SEPARATORS,
)

file = "../data/swapples/src/components/game/GameBoardComponent.js"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

    chunks = js_splitter.split_text(text)
    print(len(chunks))
    print(chunks[3])
    token_count = count_tokens(chunks[3])
    print(f"Token count for chunk 2: {token_count}")

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