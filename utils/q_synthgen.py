import json
import os
from typing import Dict, Any
from openai import OpenAI
from config import OPEN_API_KEY

class QuestionSynthesizer:
    def __init__(self, model: str = "gpt-5-mini"):
        """
        Initialize the Question Synthesizer
        
        Args:
            model: Model name to use for generation
        """
        self.model = model
        self.client = OpenAI(api_key=OPEN_API_KEY)
    
    def generate_question(self, code_content: str) -> str:
        prompt = f"""Given the following code, generate a clear and specific question that this code would be an appropriate answer to. The question should be practical and educational, focusing on what the code does, how it works, or what problem it solves.

Code:
```
{code_content}
```

Generate only the question in natural language, without any code blocks, formatting, or additional explanation. Don't make observations."""

        try:
            print(f"  → Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000,
                temperature=1,
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Clean up the generated question
            # Remove any quotes or formatting that might have been added
            generated_text = generated_text.strip('"\'`')
            
            print(f"  → Received response: {generated_text}")
            return generated_text if generated_text else "What does this code do?"
            
        except Exception as e:
            print(f"Error generating question with OpenAI: {e}")
            return "What does this code do?"
    
    def process_jsonl_file(self, input_file: str, output_file: str = None, max_entries: int = None) -> None:
        """
        Process a JSONL file and generate questions for entries with empty questions
        
        Args:
            input_file: Path to the input JSONL file
            output_file: Path to the output JSONL file (if None, will overwrite input)
            max_entries: Maximum number of entries to process (if None, process all)
        """
        if output_file is None:
            output_file = input_file
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        processed_entries = []
        total_entries = 0
        generated_questions = 0
        
        print(f"Reading from: {input_file}")
        if max_entries:
            print(f"Processing maximum {max_entries} entries")
        
        # Read and process each line
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Stop if we've reached the maximum number of entries
                if max_entries and total_entries >= max_entries:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    total_entries += 1
                    
                    # Check if question is empty or missing
                    if not entry.get("Question", "").strip():
                        answer_content = entry.get("Answer", "")
                        
                        if answer_content.strip():
                            print(f"\n📝 Processing entry {line_num}/{max_entries if max_entries else '?'}...")
                            print(f"  → Code snippet preview: {answer_content[:100].replace(chr(10), ' ')[:100]}...")
                            generated_question = self.generate_question(answer_content)
                            entry["Question"] = generated_question
                            generated_questions += 1
                            print(f"  ✅ Generated question: {generated_question}")
                        else:
                            print(f"⚠️  Skipping entry {line_num} - no answer content")
                    else:
                        print(f"ℹ️  Entry {line_num} already has a question, skipping")
                    
                    processed_entries.append(entry)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        # Write the processed entries to output file
        print(f"\n💾 Writing results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in processed_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n🎉 Processing complete!")
        print(f"📊 Total entries processed: {total_entries}")
        print(f"❓ Questions generated: {generated_questions}")
        print(f"📁 Output written to: {output_file}")

def main():
    """Main function to run the question synthesizer"""
    
    # Configuration
    INPUT_FILE = r"w:\Users\cayab\dataset-QA-prep\data\outputs\concated.jsonl"
    OUTPUT_FILE = r"w:\Users\cayab\dataset-QA-prep\data\outputs\concated_with_questions.jsonl"
    MODEL = "gpt-5-mini"
    
    # Initialize synthesizer
    synthesizer = QuestionSynthesizer(model=MODEL)
    
    # Test OpenAI connection
    try:
        test_response = synthesizer.client.models.list()
        print("✓ Successfully connected to OpenAI")
    except Exception as e:
        print(f"✗ Failed to connect to OpenAI: {e}")
        print("Please ensure your OpenAI API key is valid")
        return
    
    # Process the file
    try:
        synthesizer.process_jsonl_file(INPUT_FILE, OUTPUT_FILE, max_entries=5)
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
