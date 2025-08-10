import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjs
import json
import os

def parse_js_functions(file_path, output_path):
    """
    Parse JavaScript file using tree-sitter and extract functions as JSON
    """
    # Initialize the JavaScript language and parser
    JS_LANGUAGE = Language(tsjs.language())
    parser = Parser(JS_LANGUAGE)
    
    # Read the JavaScript file
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Parse the source code
    tree = parser.parse(bytes(source_code, 'utf8'))
    root_node = tree.root_node
    
    functions = []
    
    def extract_functions(node):
        """Recursively extract functions from AST nodes"""
        # Different types of function declarations in JavaScript
        function_types = [
            'function_declaration',
            'function_expression', 
            'arrow_function',
            'method_definition',
            'generator_function_declaration',
            'async_function_declaration'
        ]
        
        if node.type in function_types:
            # Extract function information
            func_info = {
                'type': node.type,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'start_char': node.start_byte,
                'end_char': node.end_byte,
                'text': node.text.decode('utf-8')
            }
            
            # Try to extract function name
            func_info['name'] = None
            
            if node.type == 'function_declaration' or node.type == 'async_function_declaration' or node.type == 'generator_function_declaration':
                # Regular function declaration: function name() {}
                for child in node.children:
                    if child.type == 'identifier':
                        func_info['name'] = child.text.decode('utf-8')
                        break
            elif node.type == 'method_definition':
                # Method in class or object: methodName() {}
                for child in node.children:
                    if child.type == 'property_identifier':
                        func_info['name'] = child.text.decode('utf-8')
                        break
            elif node.type == 'function_expression':
                # Function expression: const func = function() {} or function name() {}
                for child in node.children:
                    if child.type == 'identifier':
                        func_info['name'] = child.text.decode('utf-8')
                        break
            
            # For arrow functions and anonymous functions, try to get name from parent assignment
            if func_info['name'] is None and node.parent:
                parent = node.parent
                if parent.type == 'variable_declarator':
                    for child in parent.children:
                        if child.type == 'identifier':
                            func_info['name'] = child.text.decode('utf-8')
                            break
                elif parent.type == 'assignment_expression':
                    left_child = parent.children[0] if parent.children else None
                    if left_child and left_child.type == 'identifier':
                        func_info['name'] = left_child.text.decode('utf-8')
                elif parent.type == 'pair':
                    # Object property: { methodName: function() {} }
                    for child in parent.children:
                        if child.type == 'property_identifier' or child.type == 'identifier':
                            func_info['name'] = child.text.decode('utf-8')
                            break
            
            # If still no name, mark as anonymous
            if func_info['name'] is None:
                func_info['name'] = 'anonymous'
            
            functions.append(func_info)
        
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
    js_file = r"w:\Users\cayab\dataset-QA-prep\data\simple\GameBoardComponent.js"
    output_file = r"w:\Users\cayab\dataset-QA-prep\treesitter\parsed_js_functions.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Parse the JavaScript file
    functions = parse_js_functions(js_file, output_file)
    
    print(f"Parsed {len(functions)} functions from {js_file}")
    print(f"Output saved to {output_file}")
    
    # Print summary of found functions
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func.get('name', 'unnamed')} ({func['type']}) - lines {func['start_line']}-{func['end_line']}")

if __name__ == "__main__":
    main()