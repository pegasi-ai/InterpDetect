
#!pip install trulens trulens-providers-openai chromadb openai

import os
os.environ["TRULENS_OTEL_TRACING"] = "1"

from openai import OpenAI
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.core import TruSession
import numpy as np
from trulens.core import Feedback
from trulens.providers.openai import OpenAI
from trulens.apps.app import TruApp
from trulens.dashboard import run_dashboard
import time
import pandas as pd
import argparse
import sys
from dotenv import load_dotenv

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

class RAG:
    def __init__(self, model_name: str = None, context: str = None, response: str = None):
        self.model_name = model_name
        self.completion = response
        self.retrieved = context

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        return [self.retrieved]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        
        return self.completion
       
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        context_str = self.retrieve(query=query)
        completion = self.generate_completion(
            query=query, context_str=context_str
        )
        return completion

def run_trulens_evaluation(df, model_name, wait_time=20):
    """Run TruLens evaluation on the dataset"""
    print(f"Setting up TruLens with model: {model_name}")
    
    try:
        session = TruSession()
        session.reset_database()
        
        provider = OpenAI(model_engine=model_name)
        
        # Define a groundedness feedback function
        f_groundedness = (
            Feedback(
                provider.groundedness_measure_with_cot_reasons, name="Groundedness"
            )
            .on_context(collect_list=True)
            .on_output()
        )
    except Exception as e:
        print(f"Error setting up TruLens with model {model_name}: {e}")
        return df
    
    trulens_groundedness = []

    for i, row in df.iterrows():
        if i%10 == 0:
            print(f"Processing sample {i}/{len(df)}")

        try:
            context = " ".join(row['documents'])
            query = row['question']
            response = row['response']

            rag = RAG(model_name, context, response)

            tru_rag = TruApp(
                rag,
                app_name="OTEL-RAG",
                app_version=model_name,
                feedbacks=[f_groundedness],
            )

            with tru_rag as recording:
                rag.query(query)

            df_res = session.get_leaderboard()
            run_dashboard(session)
            time.sleep(wait_time)
            df_res = session.get_leaderboard()
            run_dashboard(session)
            
            trulens_groundedness.append(df_res.iloc[0]['Groundedness'])
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            trulens_groundedness.append(-1)  # Error indicator

    df['trulens_groundedness'] = trulens_groundedness
    return df

def evaluate_thresholds(df, model_name, threshold_min=0.3, threshold_max=0.85, threshold_step=0.01):
    """Evaluate different thresholds for TruLens groundedness scores"""
    print(f"Evaluating thresholds for model: {model_name}")
    
    # Generate threshold list
    thresholds = np.arange(threshold_min, threshold_max, threshold_step).tolist()
    
    best_f1 = 0
    best_threshold = threshold_min
    best_metrics = {}
    
    for threshold in thresholds:
        print(f"Threshold: {threshold:.2f}")

        tp, fp, fn = 0, 0, 0

        for _, row in df.iterrows():
            if len(row['labels']) == 0:  # no hallucination
                if row['trulens_groundedness'] < threshold:
                    fp += 1
            else: # hallucination
                if row['trulens_groundedness'] < threshold:
                    tp += 1
                else:
                    fn += 1

        if tp+fp==0 or tp+fn==0:
            continue

        p, r = tp/(tp+fp), tp/(tp+fn)

        if p+r==0:
            continue
        f1 = 2.*p*r/(p+r)
        
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
    """Main function to run the TruLens baseline evaluation"""
    parser = argparse.ArgumentParser(description='Run TruLens baseline for hallucination detection')
    parser.add_argument('--data_path', type=str, 
                       default="../../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json",
                       help='Path to the test data file')
    parser.add_argument('--model', type=str, 
                       default="gpt-4o",
                       help='OpenAI model to use for TruLens evaluation')
    parser.add_argument('--wait_time', type=int,
                       default=20,
                       help='Wait time between evaluations (seconds)')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_path', type=str,
                       default='trulens_baseline_results.json',
                       help='Path to save results')
    parser.add_argument('--threshold_min', type=float,
                       default=0.3,
                       help='Minimum threshold for evaluation')
    parser.add_argument('--threshold_max', type=float,
                       default=0.85,
                       help='Maximum threshold for evaluation')
    parser.add_argument('--threshold_step', type=float,
                       default=0.01,
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
    
    # Run TruLens evaluation
    print(f"\nRunning TruLens evaluation with model: {args.model}")
    df = run_trulens_evaluation(df, args.model, args.wait_time)
    
    # Evaluate thresholds
    result = evaluate_thresholds(df, args.model, args.threshold_min, args.threshold_max, args.threshold_step)
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



