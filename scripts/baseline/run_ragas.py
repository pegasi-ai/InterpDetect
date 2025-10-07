# %%
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI
import json
import pandas as pd
import argparse
import sys
import numpy as np
from dotenv import load_dotenv
import os

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

def run_ragas_evaluation(df, model_name):
    """Run RAGAS evaluation on the dataset"""
    print(f"Loading model: {model_name}")
    
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return df
    
    ragas_faithfulness = []

    for i, row in df.iterrows():
        if i%50 == 0:
            print(f"Processing sample {i}/{len(df)}")

        try:
            data = {
                "question": [row['question']],
                "answer": [row['response']],
                "contexts": [row['documents']]
            }
            dataset = Dataset.from_dict(data)

            # Run evaluation
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm)
            ragas_faithfulness.append(result['faithfulness'][0])
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            ragas_faithfulness.append(-1)  # Error indicator

    df['ragas_faithfulness'] = ragas_faithfulness
    return df

def evaluate_thresholds(df, model_name):
    """Evaluate different thresholds for RAGAS faithfulness scores"""
    print(f"Evaluating thresholds for model: {model_name}")
    
    # Generate threshold list
    thresholds = np.arange(0.5, 0.91, 0.05).tolist()
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        print(f"Threshold: {threshold:.2f}")

        tp, fp, fn = 0, 0, 0

        for _, row in df.iterrows():
            if len(row['labels']) == 0:  # no hallucination
                if row['ragas_faithfulness'] < threshold:
                    fp += 1
            else: # hallucination
                if row['ragas_faithfulness'] < threshold:
                    tp += 1
                else:
                    fn += 1

        p = tp/(tp+fp) if (tp+fp) > 0 else 0
        r = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2.*p*r/(p+r) if (p+r) > 0 else 0
        
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': p,
                'recall': r,
                'f1': f1
            }
    
    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best F1-score: {best_f1:.4f}")
    print("-" * 50)
    
    return best_metrics

def main():
    """Main function to run the RAGAS baseline evaluation"""
    parser = argparse.ArgumentParser(description='Run RAGAS baseline for hallucination detection')
    parser.add_argument('--data_path', type=str, 
                       default="../../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json",
                       help='Path to the test data file')
    parser.add_argument('--model', type=str, 
                       default="gpt-4o",
                       help='OpenAI model to use for RAGAS evaluation')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_path', type=str,
                       default='ragas_baseline_results.json',
                       help='Path to save results')
    parser.add_argument('--threshold_min', type=float,
                       default=0.5,
                       help='Minimum threshold for evaluation')
    parser.add_argument('--threshold_max', type=float,
                       default=0.9,
                       help='Maximum threshold for evaluation')
    parser.add_argument('--threshold_step', type=float,
                       default=0.05,
                       help='Step size for threshold evaluation')
    
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
    
    # Run RAGAS evaluation
    print(f"\nRunning RAGAS evaluation with model: {args.model}")
    df = run_ragas_evaluation(df, args.model)
    
    # Evaluate thresholds
    result = evaluate_thresholds(df, args.model)
    result['model'] = args.model
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output_path}")
    
    return df, result

if __name__ == "__main__":
    df, result = main()



