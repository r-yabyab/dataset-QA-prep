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




TEXT_TO_COUNT = """
def nextGen(self):

self.current_gen += 1
self.change_gen[self.current_gen % 3] = copy.copy(self.grid)
grid_cp = copy.copy(self.grid)

for cell in self.grid:
y, x = cell
y1 = (y - 1) % self.y_grid
y2 = (y + 1) % self.y_grid
x1 = (x - 1) % self.x_grid
x2 = (x + 1) % self.x_grid
n = self.countNeighbours(cell)

if n < 2 or n > 3:
del grid_cp[cell]
self.addchar(y + self.y_pad, x + self.x_pad, ' ')
else:
grid_cp[cell] = min(self.grid[cell] + 1, self.color_max)

for neighbour in product([y1, y, y2], [x1, x, x2]):
if not self.grid.get(neighbour):
if self.countNeighbours(neighbour) == 3:
y, x = neighbour
y = y % self.y_grid
x = x % self.x_grid
neighbour = y, x
grid_cp[neighbour] = 1

self.grid = grid_cp
"""

if __name__ == "__main__":
    print("Counting tokens for the sample text...")
    result = count_tokens(TEXT_TO_COUNT)
    print(f"Llama token count: {result['llama_token_count']}")
    print(f"GPT-2 token count: {result['gpt2_token_count']}")