# dataset-QA-prep

Scripts to prepare source code into Question, Answer dataset samples for code generation SFT. Logs filtered data into separate files.
</br>

### Workflow
- Split source code
    - Filter out unwanted data via .gitignore first, like node_modules/
    - Turn into AST then parse by function. Output into Answer fields using <b>tree-sitter</b>
- Filter out small and large samples
    - Tokenize Answers with LLM's tokenizer
    - Filter out functions > 500 tokens, < 5 chars
- Filter out duplicates
    - Code embeddings with uniXcoder 
        - https://github.com/microsoft/CodeBERT/tree/master/UniXcoder
        - Embeds code based on syntax and semantics for multiple languages
    - Compare samples via cosine_similarity
        - Can detect exact string duplicates @ 1.00, best to implement actual string checker as a fail-safe
        - Sorts filtered data from highest to threshhold, used to help determine the best cutoff to find duplicates based on language, data
- Question synthetic generation
    - gpt-5-nano generates Question based on Answer
    - Appends to Question
- Templating
    - Format jsonl to LLM's template

### Testing
- Split data into train and eval
- See if improvement between half and full dataset
#### Testing inference after SFT
- To implement pass@k tests
    - Add samples and additional tests based on output that doesn't work