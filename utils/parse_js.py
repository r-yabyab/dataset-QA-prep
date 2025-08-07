from pyjsparser import PyJsParser

parser = PyJsParser()

file = "../data/simple/GameBoardComponent.js"

with open(file, "r") as f:
    js_code = f.read()

ast = parser.parse(js_code)

# Print top-level function names
for node in ast['body']:
    if node['type'] == 'FunctionDeclaration':
        name = node['id']['name']
        print(f"Function found: {name}")