# dataset-QA-prep

For SFT code generation. Prepare data via linting and synthetic Q generation. 
</br>

Split into QA pairs
- Q synthetically generated
- Add context via RAG
- Append to A

Preprocess QA
- Static lint Qs from dataset
- Restructure into LLM specific input template

Inference after finetune
- Static lint outputs