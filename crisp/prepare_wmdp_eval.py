"""
Preparation script for WMDP-bio evaluation data.

This script downloads and formats WMDP-bio evaluation questions
(MCQ format) from HuggingFace and saves them in the format expected
by the evaluation pipeline.

Usage:
    python prepare_wmdp_eval.py
"""

import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm

from globals import DATA_PATH

def format_mcq_data(dataset, output_path):
    """
    Format dataset into MCQ JSON format expected by eval.py.
    
    Expected output format:
    {
        "questions": ["Question 1?", "Question 2?", ...],
        "answers": ["A", "B", ...],
        "choices": [
            ["Choice A", "Choice B", "Choice C", "Choice D"],
            ...
        ]
    }
    
    Args:
        dataset: HuggingFace dataset
        output_path: Path to save formatted JSON
    """
    questions = []
    answers = []
    choices = []
    
    for item in tqdm(dataset, desc="Processing questions"):
        # Handle different possible formats
        if 'question' in item:
            question = item['question']
        elif 'text' in item:
            question = item['text']
        else:
            # Try to find the question field
            question = str(item)
        
        questions.append(question)
        
        # Get answer (usually "A", "B", "C", or "D")
        if 'answer' in item:
            answer = item['answer']
        elif 'label' in item:
            # Convert label index to letter if needed
            label = item['label']
            if isinstance(label, int):
                answer = chr(65 + label)  # 0->A, 1->B, etc.
            else:
                answer = label
        else:
            answer = "A"  # Default fallback
        
        answers.append(answer)
        
        # Get choices
        if 'choices' in item:
            choice_list = item['choices']
        elif 'options' in item:
            choice_list = item['options']
        else:
            # Try to extract from A, B, C, D fields
            choice_list = []
            for letter in ['A', 'B', 'C', 'D']:
                if letter in item:
                    choice_list.append(item[letter])
                elif letter.lower() in item:
                    choice_list.append(item[letter.lower()])
        
        # Ensure we have 4 choices
        while len(choice_list) < 4:
            choice_list.append("")
        
        choices.append(choice_list[:4])
    
    # Create output format
    output_data = {
        "questions": questions,
        "answers": answers,
        "choices": choices
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved {len(questions)} questions to {output_path}")
    
    return output_data

def prepare_wmdp_bio_eval():
    """
    Prepare WMDP-bio evaluation MCQ data.
    """
    print("=" * 60)
    print("Preparing WMDP-Bio Evaluation Data")
    print("=" * 60)
    print()
    
    output_dir = os.path.join(DATA_PATH, "wmdp", "bio")
    
    # 1. WMDP-Bio MCQ
    print("[1/3] Loading WMDP-Bio MCQ dataset...")
    try:
        wmdp_bio = load_dataset("cais/wmdp", "wmdp-bio", split="test")
        print(f"Loaded {len(wmdp_bio)} questions")
        
        output_path = os.path.join(output_dir, "bio_mcq.json")
        format_mcq_data(wmdp_bio, output_path)
        
    except Exception as e:
        print(f"✗ Failed to load WMDP-bio MCQ: {e}")
        print("Creating placeholder file...")
        
        # Create placeholder with empty data
        output_path = os.path.join(output_dir, "bio_mcq.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({"questions": [], "answers": [], "choices": []}, f)
    
    print()
    
    # 2. College Biology (from MMLU)
    print("[2/3] Loading College Biology (MMLU)...")
    try:
        college_bio = load_dataset("cais/mmlu", "college_biology", split="test")
        print(f"Loaded {len(college_bio)} questions")
        
        output_path = os.path.join(output_dir, "college_bio_mcq.json")
        format_mcq_data(college_bio, output_path)
        
    except Exception as e:
        print(f"✗ Failed to load College Biology: {e}")
        print("Trying alternative dataset...")
        
        try:
            college_bio = load_dataset("tasksource/mmlu", "college_biology", split="test")
            print(f"Loaded {len(college_bio)} questions from alternative source")
            
            output_path = os.path.join(output_dir, "college_bio_mcq.json")
            format_mcq_data(college_bio, output_path)
            
        except Exception as e2:
            print(f"✗ Also failed: {e2}")
            print("Creating placeholder file...")
            
            output_path = os.path.join(output_dir, "college_bio_mcq.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({"questions": [], "answers": [], "choices": []}, f)
    
    print()
    
    # 3. High School Biology (from MMLU)
    print("[3/3] Loading High School Biology (MMLU)...")
    try:
        hs_bio = load_dataset("cais/mmlu", "high_school_biology", split="test")
        print(f"Loaded {len(hs_bio)} questions")
        
        output_path = os.path.join(output_dir, "high_school_bio_mcq.json")
        format_mcq_data(hs_bio, output_path)
        
    except Exception as e:
        print(f"✗ Failed to load High School Biology: {e}")
        print("Trying alternative dataset...")
        
        try:
            hs_bio = load_dataset("tasksource/mmlu", "high_school_biology", split="test")
            print(f"Loaded {len(hs_bio)} questions from alternative source")
            
            output_path = os.path.join(output_dir, "high_school_bio_mcq.json")
            format_mcq_data(hs_bio, output_path)
            
        except Exception as e2:
            print(f"✗ Also failed: {e2}")
            print("Creating placeholder file...")
            
            output_path = os.path.join(output_dir, "high_school_bio_mcq.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({"questions": [], "answers": [], "choices": []}, f)
    
    print()
    print("=" * 60)
    print("Evaluation data preparation complete!")
    print(f"Files saved to: {output_dir}")
    print("=" * 60)

def prepare_wmdp_cyber_eval():
    """
    Prepare WMDP-cyber evaluation MCQ data (bonus).
    """
    print("=" * 60)
    print("Preparing WMDP-Cyber Evaluation Data")
    print("=" * 60)
    print()
    
    output_dir = os.path.join(DATA_PATH, "wmdp", "cyber")
    
    # WMDP-Cyber MCQ
    print("Loading WMDP-Cyber MCQ dataset...")
    try:
        wmdp_cyber = load_dataset("cais/wmdp", "wmdp-cyber", split="test")
        print(f"Loaded {len(wmdp_cyber)} questions")
        
        output_path = os.path.join(output_dir, "cyber_mcq.json")
        format_mcq_data(wmdp_cyber, output_path)
        
    except Exception as e:
        print(f"✗ Failed to load WMDP-cyber MCQ: {e}")
        print("Creating placeholder file...")
        
        output_path = os.path.join(output_dir, "cyber_mcq.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({"questions": [], "answers": [], "choices": []}, f)
    
    print()
    
    # High School Computer Science
    print("Loading High School Computer Science (MMLU)...")
    try:
        hs_cs = load_dataset("cais/mmlu", "high_school_computer_science", split="test")
        print(f"Loaded {len(hs_cs)} questions")
        
        output_path = os.path.join(output_dir, "high_school_computer_science_mcq.json")
        format_mcq_data(hs_cs, output_path)
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        
        try:
            hs_cs = load_dataset("tasksource/mmlu", "high_school_computer_science", split="test")
            print(f"Loaded {len(hs_cs)} questions from alternative source")
            
            output_path = os.path.join(output_dir, "high_school_computer_science_mcq.json")
            format_mcq_data(hs_cs, output_path)
            
        except Exception as e2:
            print(f"✗ Also failed: {e2}")
            print("Creating placeholder file...")
            
            output_path = os.path.join(output_dir, "high_school_computer_science_mcq.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({"questions": [], "answers": [], "choices": []}, f)
    
    print()
    
    # College Computer Science
    print("Loading College Computer Science (MMLU)...")
    try:
        college_cs = load_dataset("cais/mmlu", "college_computer_science", split="test")
        print(f"Loaded {len(college_cs)} questions")
        
        output_path = os.path.join(output_dir, "college_computer_science_mcq.json")
        format_mcq_data(college_cs, output_path)
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        
        try:
            college_cs = load_dataset("tasksource/mmlu", "college_computer_science", split="test")
            print(f"Loaded {len(college_cs)} questions from alternative source")
            
            output_path = os.path.join(output_dir, "college_computer_science_mcq.json")
            format_mcq_data(college_cs, output_path)
            
        except Exception as e2:
            print(f"✗ Also failed: {e2}")
            print("Creating placeholder file...")
            
            output_path = os.path.join(output_dir, "college_computer_science_mcq.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({"questions": [], "answers": [], "choices": []}, f)
    
    print()
    print("=" * 60)
    print("Evaluation data preparation complete!")
    print(f"Files saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    # Prepare WMDP-Bio evaluation data (primary)
    prepare_wmdp_bio_eval()
    
    print("\n" + "=" * 60)
    
    # Ask if user wants to prepare cyber eval data too
    user_input = input("\nPrepare WMDP-Cyber evaluation data too? (y/N): ").lower()
    
    if user_input == 'y':
        print()
        prepare_wmdp_cyber_eval()
    else:
        print("\nSkipping WMDP-Cyber evaluation data.")
    
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)

