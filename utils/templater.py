# for llama-8b

import json
import os

def format_jsonl_to_conversations(input_file, output_file):
    """
    Convert JSONL file from Question/Answer format to conversations format.
    
    Input format: {"Question": "...", "Answer": "..."}
    Output format: {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        processed_count = 0
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Check if required fields exist
                if 'Question' not in data or 'Answer' not in data:
                    print(f"Warning: Line {line_num} missing Question or Answer field, skipping.")
                    continue
                
                conversation = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": data["Question"]
                        },
                        {
                            "role": "assistant", 
                            "content": data["Answer"]
                        }
                    ]
                }
                
                outfile.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
        
        print(f"Successfully processed {processed_count} entries.")
        print(f"Output saved to: {output_file}")

def main():
    input_file = "W:/Users/cayab/dataset-QA-prep/data/outputs/concated_with_questions.jsonl"
    output_file = "w:/Users/cayab/dataset-QA-prep/data/outputs/formatted_conversations.jsonl"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    format_jsonl_to_conversations(input_file, output_file)

if __name__ == "__main__":
    main()
