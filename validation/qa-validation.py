import json
import sys
from collections import defaultdict
from pathlib import Path

def load_jsonl_dataset(file_path):
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    dataset.append(entry)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error on line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return dataset

def validate_qa_dataset(dataset):
    """Validate Question/Answer format dataset"""
    format_errors = defaultdict(int)
    total_entries = len(dataset)
    
    print(f"Validating {total_entries} entries...")
    
    for i, ex in enumerate(dataset):
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
        
        if "Question" not in ex:
            format_errors["missing_question_key"] += 1
        if "Answer" not in ex:
            format_errors["missing_answer_key"] += 1
            
        # Check for unexpected keys
        expected_keys = {"Question", "Answer"}
        unexpected_keys = set(ex.keys()) - expected_keys
        if unexpected_keys:
            format_errors["unexpected_keys"] += 1
        
        question = ex.get("Question", None)
        if question is None:
            format_errors["question_is_null"] += 1
        elif not isinstance(question, str):
            format_errors["question_not_string"] += 1
        elif question.strip() != "":
            format_errors["question_is_empty"] += 1
        
        answer = ex.get("Answer", None)
        if answer is None:
            format_errors["answer_is_null"] += 1
        elif not isinstance(answer, str):
            format_errors["answer_not_string"] += 1
        elif len(answer.strip()) <= 15:
            format_errors["answer_too_short"] += 1
        elif answer.strip() == "":
            format_errors["answer_is_empty"] += 1
    
    return format_errors

def main():
    file_path = "../data/testing/concated_with_questions-080725-2123-mini.jsonl"
    
    print(f"Loading dataset from: {file_path}")
    dataset = load_jsonl_dataset(file_path)
    
    if not dataset:
        print("No valid entries found in dataset")
        sys.exit(1)
    
    format_errors = validate_qa_dataset(dataset)
    
    if format_errors:
        print("\nFound errors:")
        for error_type, count in format_errors.items():
            print(f"  {error_type}: {count}")
        print(f"\nTotal errors: {sum(format_errors.values())}")
    else:
        print("\nNo errors found! Dataset is valid.")
    
    print(f"Validation complete.")

if __name__ == "__main__":
    main()