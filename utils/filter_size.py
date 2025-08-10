# filters out small (chars) and large (model tokenizer)

import json
import os
from transformers import AutoTokenizer

def filter_small_answers(input_file, rejected_file, output_file, max_lines=1000):
    """
    Process a JSONL file line by line, detecting entries with Answer field < 5 characters
    or > 500 tokens using Llama tokenizer.
    Write rejected entries to rejected_small.jsonl and valid entries to a new output file.
    Process only the first max_lines lines.
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return
    
    # Initialize Llama tokenizer
    print("Loading Llama tokenizer...")
    llama_tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    print("Tokenizer loaded successfully.")
    
    valid_count = 0
    rejected_small_count = 0
    rejected_large_count = 0
    lines_processed = 0
    
    # Read the original file and filter lines
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(rejected_file, 'w', encoding='utf-8') as reject_file:
            with open(output_file, 'w', encoding='utf-8') as output_f:
                for line_num, line in enumerate(infile, 1):
                    # Stop processing after max_lines
                    if lines_processed >= max_lines:
                        print(f"Reached maximum lines limit ({max_lines}). Stopping processing.")
                        break
                    
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    lines_processed += 1
                    
                    try:
                        # Parse the JSON line
                        data = json.loads(line)
                        
                        # Check if Answer field exists and its length
                        answer = data.get('Answer', '')
                        
                        # Check for small answers (< 5 characters)
                        if len(answer) < 5:
                            # Write to rejected file
                            reject_file.write(line + '\n')
                            rejected_small_count += 1
                            print(f"Line {line_num}: Rejected (Answer too small - length: {len(answer)})")
                        else:
                            # Check for large answers (> 500 tokens)
                            tokens = llama_tokenizer.encode(answer)
                            token_count = len(tokens)
                            
                            if token_count > 500:
                                # Write to rejected file
                                reject_file.write(line + '\n')
                                rejected_large_count += 1
                                print(f"Line {line_num}: Rejected (Answer too large - tokens: {token_count})")
                            else:
                                # Write to output file
                                output_f.write(line + '\n')
                                valid_count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        # Keep malformed lines in the output file
                        output_f.write(line + '\n')
                        valid_count += 1
    
    print(f"\nProcessing complete:")
    print(f"- Lines processed: {lines_processed}")
    print(f"- Rejected entries (too small): {rejected_small_count}")
    print(f"- Rejected entries (too large): {rejected_large_count}")
    print(f"- Total rejected entries: {rejected_small_count + rejected_large_count}")
    print(f"- Valid entries: {valid_count}")
    print(f"- Rejected entries saved to: {rejected_file}")
    print(f"- Valid entries saved to: {output_file}")

if __name__ == "__main__":
    # Process the parsed_java_functions.jsonl file (first 1000 lines only)
    input_file = r"W:\Users\cayab\dataset-QA-prep\data\java-data\parsed_java_functions.jsonl"
    rejected_file = r"w:\Users\cayab\dataset-QA-prep\data\java-data\rejected_small.jsonl"
    output_file = r"w:\Users\cayab\dataset-QA-prep\data\java-data\parsed_java_functions_removesmall.jsonl"
    
    print(f"Processing file: {input_file}")
    print(f"Rejected entries will be saved to: {rejected_file}")
    print(f"Valid entries will be saved to: {output_file}")
    print(f"Maximum lines to process: 1000")
    print("-" * 50)
    
    filter_small_answers(input_file, rejected_file, output_file, max_lines=1000)