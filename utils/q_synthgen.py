import openai
import json
import os
from typing import List, Dict, Optional
import time

class QuestionSynthGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Question Synthesis Generator
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
        """
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
        if not openai.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def generate_question_from_answer(self, answer: str, context: str = "", programming_language: str = "", difficulty: str = "medium") -> str:
        """
        Generate a question based on the provided answer
        
        Args:
            answer: The code answer to generate a question for
            context: Additional context about the problem domain
            programming_language: The programming language of the answer
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            Generated question string
        """
        
        prompt = f"""You are an expert programming instructor. Given the following code answer, generate a clear, specific programming question that would naturally lead to this solution.

Programming Language: {programming_language if programming_language else "Auto-detect"}
Difficulty Level: {difficulty}
Context: {context if context else "General programming problem"}

Code Answer:
```
{answer}
```

Requirements for the question:
1. Be specific and clear about what needs to be implemented
2. Include any necessary constraints or requirements
3. Specify input/output format if applicable
4. Make it challenging but fair for the given difficulty level
5. Don't reveal the exact solution approach

Generate only the question text, no additional commentary:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # You can change to gpt-4 if needed
                messages=[
                    {"role": "system", "content": "You are an expert programming instructor who creates clear, specific questions from code solutions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating question: {e}")
            return None
    
    def process_single_answer(self, answer: str, **kwargs) -> Dict[str, str]:
        """
        Process a single answer and return question-answer pair
        
        Args:
            answer: The code answer
            **kwargs: Additional parameters for question generation
            
        Returns:
            Dictionary with 'question' and 'answer' keys
        """
        question = self.generate_question_from_answer(answer, **kwargs)
        
        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "programming_language": kwargs.get("programming_language", ""),
                "difficulty": kwargs.get("difficulty", "medium"),
                "context": kwargs.get("context", "")
            }
        }
    
    def process_answers_from_file(self, file_path: str, output_path: str = None, **kwargs) -> List[Dict[str, str]]:
        """
        Process answers from a file and generate questions
        
        Args:
            file_path: Path to file containing answers (one per line or JSON format)
            output_path: Path to save the generated Q&A pairs
            **kwargs: Additional parameters for question generation
            
        Returns:
            List of question-answer dictionaries
        """
        qa_pairs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        answers = data
                    else:
                        answers = [data]
                else:
                    answers = [line.strip() for line in f.readlines() if line.strip()]
            
            for i, answer in enumerate(answers):
                print(f"Processing answer {i+1}/{len(answers)}...")
                
                if isinstance(answer, dict):
                    # If answer is a dict, extract the actual answer content
                    answer_text = answer.get('answer', answer.get('code', str(answer)))
                    context = answer.get('context', kwargs.get('context', ''))
                    lang = answer.get('language', kwargs.get('programming_language', ''))
                    difficulty = answer.get('difficulty', kwargs.get('difficulty', 'medium'))
                else:
                    answer_text = str(answer)
                    context = kwargs.get('context', '')
                    lang = kwargs.get('programming_language', '')
                    difficulty = kwargs.get('difficulty', 'medium')
                
                qa_pair = self.process_single_answer(
                    answer_text,
                    context=context,
                    programming_language=lang,
                    difficulty=difficulty
                )
                
                if qa_pair['question']:  # Only add if question generation was successful
                    qa_pairs.append(qa_pair)
                
                # Add a small delay to respect API rate limits
                time.sleep(1)
        
        except Exception as e:
            print(f"Error processing file: {e}")
            return qa_pairs
        
        # Save to output file if specified
        if output_path:
            self.save_qa_pairs(qa_pairs, output_path)
        
        return qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict[str, str]], output_path: str):
        """
        Save question-answer pairs to a file
        
        Args:
            qa_pairs: List of Q&A dictionaries
            output_path: Path to save the file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def append_questions_to_existing(self, qa_pairs: List[Dict[str, str]], existing_file: str):
        """
        Append new Q&A pairs to an existing file
        
        Args:
            qa_pairs: New Q&A pairs to append
            existing_file: Path to existing file
        """
        try:
            # Load existing data
            if os.path.exists(existing_file):
                with open(existing_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # Append new pairs
            existing_data.extend(qa_pairs)
            
            # Save updated data
            self.save_qa_pairs(existing_data, existing_file)
            
        except Exception as e:
            print(f"Error appending to file: {e}")


# Example usage functions
def example_usage():
    """Example of how to use the QuestionSynthGenerator"""
    
    # Initialize generator (you'll provide the API key)
    generator = QuestionSynthGenerator()  # Will use OPENAI_API_KEY env variable
    
    # Example 1: Generate question from a single answer
    code_answer = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    qa_pair = generator.process_single_answer(
        code_answer,
        programming_language="Python",
        difficulty="medium",
        context="Recursive algorithms"
    )
    
    print("Generated Q&A pair:")
    print(f"Q: {qa_pair['question']}")
    print(f"A: {qa_pair['answer']}")
    
    # Example 2: Process multiple answers from a file
    # qa_pairs = generator.process_answers_from_file(
    #     "answers.txt",
    #     output_path="generated_qa.json",
    #     programming_language="Python",
    #     difficulty="medium"
    # )


if __name__ == "__main__":
    # Set your OpenAI API key here or as environment variable
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    example_usage()
