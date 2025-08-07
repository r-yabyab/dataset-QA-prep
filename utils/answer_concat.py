import os
import json
from pathlib import Path

def concatenate_jsonl_files(data_dir="../data", output_file="../concated.jsonl"):
    """
    Concatenate all .jsonl files found in the data directory into a single .jsonl file
    
    Args:
        data_dir: Directory to search for .jsonl files
        output_file: Output file path for the concatenated .jsonl
    """
    data_path = Path(data_dir).resolve()
    output_path = Path(output_file).resolve()
    
    print(f"Searching for .jsonl files in: {data_path}")
    print(f"Output file: {output_path}")
    
    # Find all .jsonl files recursively
    jsonl_files = list(data_path.rglob("*.jsonl"))
    
    if not jsonl_files:
        print("No .jsonl files found!")
        return 0
    
    print(f"Found {len(jsonl_files)} .jsonl files:")
    for file in jsonl_files:
        print(f"  - {file.relative_to(data_path)}")
    
    total_lines = 0
    
    # Open output file for writing
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Process each .jsonl file
        for jsonl_file in jsonl_files:
            print(f"Processing: {jsonl_file.relative_to(data_path)}")
            
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as infile:
                    file_lines = 0
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Validate that it's valid JSON
                            try:
                                json.loads(line)
                                outfile.write(line + '\n')
                                file_lines += 1
                                total_lines += 1
                            except json.JSONDecodeError as e:
                                print(f"  Warning: Invalid JSON line in {jsonl_file}: {e}")
                    
                    print(f"  Added {file_lines} lines")
                    
            except Exception as e:
                print(f"  Error reading {jsonl_file}: {e}")
    
    print(f"\nConcatenation complete!")
    print(f"Total lines written: {total_lines}")
    print(f"Output saved to: {output_path}")
    
    return total_lines

if __name__ == "__main__":
    concatenate_jsonl_files()
