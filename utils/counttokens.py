from transformers import AutoTokenizer

llama_tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # GPT-3.5 uses cl100k_base tokenizer but gpt2 tokenizer is close

def count_tokens(text):
    llama_tokens = llama_tokenizer.encode(text, add_special_tokens=False)
    gpt2_tokens = gpt_tokenizer.encode(text, add_special_tokens=False)
    return {
        "llama_token_count": len(llama_tokens),
        "gpt2_token_count": len(gpt2_tokens),
    }
