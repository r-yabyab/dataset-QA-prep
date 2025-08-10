import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import json
import os

def parse_java_functions(file_path, output_path):
    """
    Parse Java file using tree-sitter and extract functions/methods as JSON
    """
    # Initialize the Java language and parser
    JAVA_LANGUAGE = Language(tsjava.language())
    parser = Parser(JAVA_LANGUAGE)
    
    # Read the Java file
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Parse the source code
    tree = parser.parse(bytes(source_code, 'utf8'))
    root_node = tree.root_node
    
    functions = []
    
    def extract_functions(node):
        """Recursively extract functions/methods from AST nodes"""
        if node.type == 'method_declaration' or node.type == 'constructor_declaration':
            # Extract method information
            method_info = {
                'type': node.type,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'start_char': node.start_byte,
                'end_char': node.end_byte,
                'text': node.text.decode('utf-8')
            }
            
            # Try to extract method name
            for child in node.children:
                if child.type == 'identifier':
                    method_info['name'] = child.text.decode('utf-8')
                    break
            
            # Extract modifiers and annotations
            modifiers = []
            annotations = []
            
            for child in node.children:
                if child.type == 'modifiers':
                    for modifier_child in child.children:
                        if modifier_child.type == 'annotation':
                            annotations.append(modifier_child.text.decode('utf-8'))
                        else:
                            modifiers.append(modifier_child.text.decode('utf-8'))
            
            method_info['modifiers'] = modifiers
            method_info['annotations'] = annotations
            
            functions.append(method_info)
        
        # Recursively process child nodes
        for child in node.children:
            extract_functions(child)
    
    # Extract functions from the parsed tree
    extract_functions(root_node)
    
    # Write to JSON file (one JSON object per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for func in functions:
            f.write(json.dumps(func) + '\n')
    
    return functions

def main():
    # Define file paths
    java_file = r"w:\Users\cayab\dataset-QA-prep\data\simple\GameController.java"
    output_file = r"w:\Users\cayab\dataset-QA-prep\data\outputs\parsed_functions.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Parse the Java file
    functions = parse_java_functions(java_file, output_file)
    
    print(f"Parsed {len(functions)} functions/methods from {java_file}")
    print(f"Output saved to {output_file}")
    
    # Print summary of found functions
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func.get('name', 'unnamed')} ({func['type']}) - lines {func['start_line']}-{func['end_line']}")

if __name__ == "__main__":
    main()