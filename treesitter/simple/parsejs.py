import tree_sitter_javascript as tsjava
from tree_sitter import Language, Parser, Node
import os
import json

def create_js_parser():
    """Create and return a JavaScript parser"""
    JS_LANGUAGE = Language(tsjava.language())
    parser = Parser(JS_LANGUAGE)
    return parser

def parse_js_file(file_path):
    """Parse a JavaScript file and return the syntax tree"""
    parser = create_js_parser()
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Parse the code
    tree = parser.parse(bytes(source_code, 'utf8'))
    return tree, source_code

def print_tree(node, source_code, depth=0):
    """Recursively print the syntax tree"""
    indent = "  " * depth
    node_text = source_code[node.start_byte:node.end_byte]
    
    # Truncate long text for readability
    if len(node_text) > 50:
        node_text = node_text[:47] + "..."
    
    # Replace newlines for cleaner output
    node_text = node_text.replace('\n', '\\n')
    
    print(f"{indent}{node.type}: {repr(node_text)}")
    
    for child in node.children:
        print_tree(child, source_code, depth + 1)

def find_nodes_by_type(node, node_type):
    """Find all nodes of a specific type"""
    results = []
    
    if node.type == node_type:
        results.append(node)
    
    for child in node.children:
        results.extend(find_nodes_by_type(child, node_type))
    
    return results

def extract_functions(tree, source_code):
    """Extract function declarations and expressions with their code"""
    functions = []
    
    # Find function declarations
    function_declarations = find_nodes_by_type(tree.root_node, "function_declaration")
    for func in function_declarations:
        name_node = func.child_by_field_name("name")
        if name_node:
            func_code = source_code[func.start_byte:func.end_byte]
            functions.append({
                'type': 'function_declaration',
                'name': source_code[name_node.start_byte:name_node.end_byte],
                'start_line': func.start_point[0] + 1,
                'end_line': func.end_point[0] + 1,
                'start_byte': func.start_byte,
                'end_byte': func.end_byte,
                'code': func_code
            })
    
    # Find arrow functions and function expressions
    arrow_functions = find_nodes_by_type(tree.root_node, "arrow_function")
    for func in arrow_functions:
        func_code = source_code[func.start_byte:func.end_byte]
        # Try to find the variable name if it's assigned
        parent = func.parent
        name = "anonymous"
        if parent and parent.type == "variable_declarator":
            name_node = parent.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte]
        
        functions.append({
            'type': 'arrow_function',
            'name': name,
            'start_line': func.start_point[0] + 1,
            'end_line': func.end_point[0] + 1,
            'start_byte': func.start_byte,
            'end_byte': func.end_byte,
            'code': func_code
        })
    
    return functions

def extract_imports(tree, source_code):
    """Extract import statements"""
    imports = []
    import_statements = find_nodes_by_type(tree.root_node, "import_statement")
    
    for imp in import_statements:
        import_text = source_code[imp.start_byte:imp.end_byte]
        imports.append({
            'statement': import_text,
            'line': imp.start_point[0] + 1
        })
    
    return imports

def extract_variables(tree, source_code):
    """Extract variable declarations"""
    variables = []
    var_declarations = find_nodes_by_type(tree.root_node, "variable_declaration")
    
    for var_decl in var_declarations:
        declarators = find_nodes_by_type(var_decl, "variable_declarator")
        for declarator in declarators:
            name_node = declarator.child_by_field_name("name")
            if name_node:
                variables.append({
                    'name': source_code[name_node.start_byte:name_node.end_byte],
                    'line': declarator.start_point[0] + 1,
                    'type': var_decl.children[0].text.decode('utf-8')  # const, let, var
                })
    
    return variables

def analyze_react_component(tree, source_code):
    """Analyze React component specific patterns"""
    analysis = {
        'component_name': None,
        'hooks': [],
        'jsx_elements': [],
        'props': []
    }
    
    # Find the main function/component
    functions = extract_functions(tree, source_code)
    if functions:
        # Assume the first function is the main component
        analysis['component_name'] = functions[0]['name']
    
    # Find React hooks (useState, useEffect, etc.)
    call_expressions = find_nodes_by_type(tree.root_node, "call_expression")
    for call in call_expressions:
        function_node = call.child_by_field_name("function")
        if function_node and function_node.type == "identifier":
            func_name = source_code[function_node.start_byte:function_node.end_byte]
            if func_name.startswith('use'):
                analysis['hooks'].append({
                    'name': func_name,
                    'line': call.start_point[0] + 1
                })
    
    # Find JSX elements
    jsx_elements = find_nodes_by_type(tree.root_node, "jsx_element")
    jsx_self_closing = find_nodes_by_type(tree.root_node, "jsx_self_closing_element")
    
    for jsx in jsx_elements + jsx_self_closing:
        opening_element = jsx.children[0] if jsx.children else jsx
        if opening_element:
            name_node = opening_element.children[1] if len(opening_element.children) > 1 else None
            if name_node:
                tag_name = source_code[name_node.start_byte:name_node.end_byte]
                analysis['jsx_elements'].append({
                    'tag': tag_name,
                    'line': jsx.start_point[0] + 1
                })
    
    return analysis

def extract_code_chunks(tree, source_code):
    """Extract code into logical chunks (functions, classes, top-level statements)"""
    chunks = []
    
    # Get all top-level statements
    program_node = tree.root_node
    
    for child in program_node.children:
        chunk_info = {
            'type': child.type,
            'start_line': child.start_point[0] + 1,
            'end_line': child.end_point[0] + 1,
            'start_byte': child.start_byte,
            'end_byte': child.end_byte,
            'code': source_code[child.start_byte:child.end_byte]
        }
        
        # Add specific information based on node type
        if child.type == "function_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                chunk_info['name'] = source_code[name_node.start_byte:name_node.end_byte]
                chunk_info['chunk_category'] = 'function'
        
        elif child.type == "variable_declaration":
            # Check if this contains a function
            declarators = find_nodes_by_type(child, "variable_declarator")
            for declarator in declarators:
                value_node = declarator.child_by_field_name("value")
                if value_node and value_node.type in ["arrow_function", "function_expression"]:
                    name_node = declarator.child_by_field_name("name")
                    if name_node:
                        chunk_info['name'] = source_code[name_node.start_byte:name_node.end_byte]
                        chunk_info['chunk_category'] = 'function'
                        break
            
            if 'chunk_category' not in chunk_info:
                chunk_info['chunk_category'] = 'variable'
        
        elif child.type == "import_statement":
            chunk_info['chunk_category'] = 'import'
            
        elif child.type == "export_statement":
            chunk_info['chunk_category'] = 'export'
            
        elif child.type == "class_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                chunk_info['name'] = source_code[name_node.start_byte:name_node.end_byte]
            chunk_info['chunk_category'] = 'class'
            
        else:
            chunk_info['chunk_category'] = 'statement'
        
        chunks.append(chunk_info)
    
    return chunks

def extract_methods_from_chunks(chunks):
    """Extract just the method/function chunks"""
    methods = []
    for chunk in chunks:
        if chunk['chunk_category'] == 'function':
            methods.append(chunk)
    return methods

def extract_codesearchnet_format(tree, source_code):
    """Extract functions in CodeSearchNet format - just the code without comments"""
    functions = []
    
    # Find function declarations
    function_declarations = find_nodes_by_type(tree.root_node, "function_declaration")
    for func in function_declarations:
        name_node = func.child_by_field_name("name")
        if name_node:
            func_name = source_code[name_node.start_byte:name_node.end_byte]
            func_code = source_code[func.start_byte:func.end_byte]
            
            # Extract docstring if present (JSDoc or regular comments before function)
            docstring = extract_function_docstring(func, source_code, tree.root_node)
            
            # Clean code - remove comments but keep structure
            clean_code = remove_comments_from_code(func_code)
            
            functions.append({
                'name': func_name,
                'code': clean_code.strip(),
                'docstring': docstring.strip() if docstring else "",
                'type': 'function_declaration',
                'start_line': func.start_point[0] + 1,
                'end_line': func.end_point[0] + 1,
                'tokens': estimate_token_count(clean_code)
            })
    
    # Find arrow functions with variable assignment
    variable_declarations = find_nodes_by_type(tree.root_node, "variable_declaration")
    for var_decl in variable_declarations:
        declarators = find_nodes_by_type(var_decl, "variable_declarator")
        for declarator in declarators:
            value_node = declarator.child_by_field_name("value")
            if value_node and value_node.type in ["arrow_function", "function_expression"]:
                name_node = declarator.child_by_field_name("name")
                if name_node:
                    func_name = source_code[name_node.start_byte:name_node.end_byte]
                    func_code = source_code[value_node.start_byte:value_node.end_byte]
                    
                    # Extract docstring (comment before variable declaration)
                    docstring = extract_function_docstring(var_decl, source_code, tree.root_node)
                    
                    # Clean code
                    clean_code = remove_comments_from_code(func_code)
                    
                    functions.append({
                        'name': func_name,
                        'code': clean_code.strip(),
                        'docstring': docstring.strip() if docstring else "",
                        'type': 'arrow_function',
                        'start_line': value_node.start_point[0] + 1,
                        'end_line': value_node.end_point[0] + 1,
                        'tokens': estimate_token_count(clean_code)
                    })
    
    return functions

def extract_function_docstring(func_node, source_code, root_node):
    """Extract JSDoc or comment associated with a function"""
    # Look for comments before the function
    func_start_line = func_node.start_point[0]
    
    # Get all comment nodes in the file
    comment_nodes = find_nodes_by_type(root_node, "comment")
    
    # Find the closest comment before this function
    closest_comment = None
    closest_distance = float('inf')
    
    for comment in comment_nodes:
        comment_end_line = comment.end_point[0]
        if comment_end_line < func_start_line:
            distance = func_start_line - comment_end_line
            if distance < closest_distance and distance <= 2:  # Within 2 lines
                closest_distance = distance
                closest_comment = comment
    
    if closest_comment:
        comment_text = source_code[closest_comment.start_byte:closest_comment.end_byte]
        # Clean up JSDoc style comments
        return clean_jsdoc_comment(comment_text)
    
    return ""

def clean_jsdoc_comment(comment_text):
    """Clean JSDoc style comments to extract just the description"""
    # Remove /** and */ and //
    cleaned = comment_text.strip()
    
    if cleaned.startswith('/**') and cleaned.endswith('*/'):
        cleaned = cleaned[3:-2]
    elif cleaned.startswith('/*') and cleaned.endswith('*/'):
        cleaned = cleaned[2:-2]
    elif cleaned.startswith('//'):
        cleaned = cleaned[2:]
    
    # Remove leading * from each line
    lines = []
    for line in cleaned.split('\n'):
        line = line.strip()
        if line.startswith('* '):
            line = line[2:]
        elif line.startswith('*'):
            line = line[1:]
        if line.strip():
            lines.append(line.strip())
    
    return ' '.join(lines)

def remove_comments_from_code(code):
    """Remove comments from JavaScript code while preserving structure"""
    lines = code.split('\n')
    cleaned_lines = []
    in_multiline_comment = False
    
    for line in lines:
        cleaned_line = ""
        i = 0
        while i < len(line):
            if not in_multiline_comment:
                # Check for single line comment
                if i < len(line) - 1 and line[i:i+2] == '//':
                    break  # Rest of line is comment
                # Check for multiline comment start
                elif i < len(line) - 1 and line[i:i+2] == '/*':
                    in_multiline_comment = True
                    i += 2
                    continue
                else:
                    cleaned_line += line[i]
            else:
                # Look for multiline comment end
                if i < len(line) - 1 and line[i:i+2] == '*/':
                    in_multiline_comment = False
                    i += 2
                    continue
            i += 1
        
        # Keep the line if it has non-whitespace content
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line.rstrip())
        elif not cleaned_line.strip() and cleaned_lines:  # Preserve some spacing
            cleaned_lines.append("")
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

def estimate_token_count(code):
    """Rough estimate of token count for code"""
    # Simple approximation: split by whitespace and common delimiters
    import re
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return len(tokens)

def filter_functions_by_token_limit(functions, max_tokens=500):
    """Filter functions to only include those under the token limit"""
    return [func for func in functions if func['tokens'] <= max_tokens]

def main():
    """Main function to demonstrate parsing and CodeSearchNet-style extraction"""
    # Path to the GameBoardComponent.js file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'data', 'simple', 'GameBoardComponent.js')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Parse the file
    tree, source_code = parse_js_file(file_path)
    
    # Extract functions in CodeSearchNet format
    print("Extracting functions in CodeSearchNet format...")
    codesearchnet_functions = extract_codesearchnet_format(tree, source_code)
    
    # Filter by token limit (like CodeSearchNet)
    max_tokens = 500
    filtered_functions = filter_functions_by_token_limit(codesearchnet_functions, max_tokens)
    
    # Extract code chunks for comparison
    chunks = extract_code_chunks(tree, source_code)
    methods = extract_methods_from_chunks(chunks)
    
    # Extract other analysis data
    imports = extract_imports(tree, source_code)
    functions = extract_functions(tree, source_code)
    variables = extract_variables(tree, source_code)
    react_analysis = analyze_react_component(tree, source_code)
    
    # Count JSX elements
    jsx_counts = {}
    for jsx in react_analysis['jsx_elements']:
        jsx_counts[jsx['tag']] = jsx_counts.get(jsx['tag'], 0) + 1
    
    # Create the JSON output structure
    analysis_result = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "codesearchnet_format": {
            "total_functions": len(codesearchnet_functions),
            "functions_under_500_tokens": len(filtered_functions),
            "functions": codesearchnet_functions,
            "filtered_functions": filtered_functions
        },
        "code_chunks": {
            "total_chunks": len(chunks),
            "chunks": chunks,
            "methods_only": methods
        },
        "imports": imports,
        "functions": functions,
        "variables": variables,
        "react_component": {
            "component_name": react_analysis['component_name'],
            "hooks": react_analysis['hooks'],
            "jsx_elements": react_analysis['jsx_elements'],
            "jsx_element_counts": jsx_counts,
            "total_hooks": len(react_analysis['hooks']),
            "total_jsx_elements": len(react_analysis['jsx_elements'])
        },
        "summary": {
            "total_imports": len(imports),
            "total_functions": len(functions),
            "total_variables": len(variables),
            "total_hooks": len(react_analysis['hooks']),
            "unique_jsx_elements": len(jsx_counts),
            "total_code_chunks": len(chunks),
            "function_chunks": len(methods),
            "codesearchnet_functions": len(codesearchnet_functions),
            "functions_under_token_limit": len(filtered_functions)
        }
    }
    
    # Output to JSON file
    output_file = os.path.join(os.path.dirname(__file__), 'analysis_output.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    # Also save CodeSearchNet format functions to separate files
    codesearchnet_dir = os.path.join(os.path.dirname(__file__), "codesearchnet_functions")
    if not os.path.exists(codesearchnet_dir):
        os.makedirs(codesearchnet_dir)
    
    for i, func in enumerate(filtered_functions):
        filename = f"{i+1:02d}_{func['name']}.js"
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        filepath = os.path.join(codesearchnet_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(func['code'])
    
    print(f"Analysis complete! Output saved to: {output_file}")
    print(f"CodeSearchNet functions saved to: {codesearchnet_dir}")
    print(f"Component: {react_analysis['component_name']}")
    print(f"Total functions found: {len(codesearchnet_functions)}")
    print(f"Functions under {max_tokens} tokens: {len(filtered_functions)}")
    
    # Print CodeSearchNet format summary
    print("\nCodeSearchNet format functions:")
    for i, func in enumerate(filtered_functions):
        docstring_preview = func['docstring'][:50] + "..." if len(func['docstring']) > 50 else func['docstring']
        print(f"  {i+1}. {func['name']} ({func['tokens']} tokens) - {docstring_preview}")
    
    return analysis_result

if __name__ == "__main__":
    main()