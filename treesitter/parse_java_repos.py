import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import json
import os
import glob
from pathlib import Path

def parse_java_functions(file_path):
    """
    Parse Java file using tree-sitter and extract functions/methods
    Returns a list of function dictionaries
    """
    try:
        # Initialize the Java language and parser
        JAVA_LANGUAGE = Language(tsjava.language())
        parser = Parser(JAVA_LANGUAGE)
        
        # Read the Java file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code = f.read()
        
        # Parse the source code
        tree = parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node
        
        functions = []
        
        def extract_functions(node):
            """Recursively extract functions/methods from AST nodes"""
            if node.type == 'method_declaration' or node.type == 'constructor_declaration':
                # Extract just the method content in the specified format
                method_content = node.text.decode('utf-8')
                
                # Create the formatted entry
                method_entry = {
                    "Question": "",
                    "Answer": method_content
                }
                
                functions.append(method_entry)
            
            # Recursively process child nodes
            for child in node.children:
                extract_functions(child)
        
        # Extract functions from the parsed tree
        extract_functions(root_node)
        
        return functions
        
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return []

def find_java_files(base_dir):
    """
    Find all Java files in the directory and its subdirectories
    """
    java_files = []
    base_path = Path(base_dir)
    
    # Walk through all subdirectories
    for java_file in base_path.rglob("*.java"):
        java_files.append(str(java_file))
    
    return java_files

def main():
    # Base directory containing repositories
    base_dir = r"W:\Users\cayab\dataset-QA-prep\data\java-data"
    output_file = r"W:\Users\cayab\dataset-QA-prep\data\java-data\parsed_java_functions.jsonl"
    
    print(f"Searching for Java files in: {base_dir}")
    
    # Find all Java files in all repositories
    java_files = find_java_files(base_dir)
    
    print(f"Found {len(java_files)} Java files")
    
    # Parse all Java files and write to JSONL
    total_functions = 0
    processed_files = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, java_file in enumerate(java_files, 1):
            print(f"Processing {i}/{len(java_files)}: {java_file}")
            
            # Parse functions from the current file
            functions = parse_java_functions(java_file)
            
            if functions:
                processed_files += 1
                total_functions += len(functions)
                
                # Write each function as a separate line in JSONL format
                for func in functions:
                    f.write(json.dumps(func) + '\n')
    
    print(f"\nCompleted!")
    print(f"Processed {processed_files} files")
    print(f"Extracted {total_functions} functions/methods")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
