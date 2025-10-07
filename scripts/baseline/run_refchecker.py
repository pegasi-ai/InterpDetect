# %%
#!pip install refchecker
# !pip install refchecker
# !python -m spacy download en_core_web_sm


import pandas as pd
from dotenv import load_dotenv
import os
from refchecker import LLMExtractor, LLMChecker
import argparse
import sys

def load_and_balance_data(file_path):
    """Load data and balance positive/negative samples"""
    df = pd.read_json(file_path, lines=False)
    
    pos, neg = [], []

    for _, row in df.iterrows():
        if len(row["labels"]) == 0:
            neg.append(row)
        else:
            pos.append(row)

    min_len = min(len(pos), len(neg))
    df = pd.DataFrame(pos[0:min_len]+neg[0:min_len])
    
    print(f"Loaded {len(df)} samples (balanced)")
    return df

def run_refchecker_evaluation(df, model_name, batch_size=8):
    """Run RefChecker evaluation on the dataset"""
    print(f"Loading RefChecker with model: {model_name}")
    
    try:
        extractor = LLMExtractor(model=model_name, batch_size=batch_size)
        checker = LLMChecker(model=model_name, batch_size=batch_size)
    except Exception as e:
        print(f"Error loading RefChecker with model {model_name}: {e}")
        return df
    
    # Prepare data
    questions = df['question'].tolist()
    responses = df['response'].tolist()
    
    references = []
    for _, row in df.iterrows():
        doc = " ".join(row["documents"])
        references.append(doc)
    
    print("Extracting claims...")
    try:
        extraction_results = extractor.extract(
            batch_responses=responses,
            batch_questions=questions,
            max_new_tokens=1000
        )
        
        batch_claims = [[c.content for c in res.claims] for res in extraction_results]
        
        print("Checking claims against references...")
        batch_labels = checker.check(
            batch_claims=batch_claims,
            batch_references=references,
            batch_questions=questions,
            max_reference_segment_length=0
        )
        
        # Process results
        refchecker_entailment = []
        for labels in batch_labels:
            if 'Contradiction' in str(labels):
                refchecker_entailment.append(1)
            else:
                refchecker_entailment.append(0)
        
        df['refchecker_entailment'] = refchecker_entailment
        
    except Exception as e:
        print(f"Error during RefChecker evaluation: {e}")
        # Add default values in case of error
        df['refchecker_entailment'] = [-1] * len(df)
    
    return df

def evaluate(df, model_name):
    """Evaluate the model performance"""
    tp, fp, fn = 0, 0, 0
    
    for _, row in df.iterrows():
        if len(row['labels']) == 0:  # no hallucination
            if row['refchecker_entailment'] == 1:
                fp += 1
        else: # hallucination
            if row['refchecker_entailment'] == 1:
                tp += 1
            else:
                fn += 1

    p = tp/(tp+fp) if (tp+fp) > 0 else 0
    r = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2.*p*r/(p+r) if (p+r) > 0 else 0
    
    print(f"Model: {model_name}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 50)
    
    return {
        'model': model_name,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': p,
        'recall': r,
        'f1': f1
    }

def main():
    """Main function to run the RefChecker baseline evaluation"""
    parser = argparse.ArgumentParser(description='Run RefChecker baseline for hallucination detection')
    parser.add_argument('--data_path', type=str, 
                       default="../../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json",
                       help='Path to the test data file')
    parser.add_argument('--model', type=str, 
                       default="gpt-4o",
                       help='OpenAI model to use for RefChecker evaluation')
    parser.add_argument('--batch_size', type=int,
                       default=8,
                       help='Batch size for RefChecker processing')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_path', type=str,
                       default='refchecker_baseline_results.json',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Load and balance data
    print(f"Loading data from: {args.data_path}")
    df = load_and_balance_data(args.data_path)
    
    # Run RefChecker evaluation
    print(f"\nRunning RefChecker evaluation with model: {args.model}")
    df = run_refchecker_evaluation(df, args.model, args.batch_size)
    
    # Evaluate results
    result = evaluate(df, args.model)
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output_path}")
    
    return df, result

if __name__ == "__main__":
    df, result = main()



