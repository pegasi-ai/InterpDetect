# %%
import pandas as pd
from dotenv import load_dotenv
import os
from groq import Groq
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


def generate_judge_prompt(context: str, question: str, response: str) -> str:
    return f"""You are an expert fact-checker. Given a context, a question, and a response, your task is to determine if the response is faithful to the context.

        Context:
        {context}

        Question:
        {question}

        Response:
        {response}

        Is the response supported and grounded in the context above? Answer "Yes" or "No", and provide a short reason if the answer is "No". Be concise and objective.
        """


def llm_as_a_judge(df, model_name, client):
    """Run LLM-as-a-judge evaluation on the dataset"""
    is_hallucinated = []
    results = []

    for i, row in df.iterrows():
        if i%50 == 0:
            print(f"Processing sample {i}/{len(df)}")

        docs = list(row['documents'])
        doc = " ".join(docs)
        question = row['question']
        response = row['response']
        prompt = generate_judge_prompt(doc, question, response)
        messages=[
            {"role": "user", "content": prompt}]

        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
            # response_format={"type": "json_object"},
            # temperature=0.3,
            # max_tokens=512,     
            )

            result = chat_completion.choices[0].message.content
            
            results.append(result)

            if "yes" in result.lower():
                is_hallucinated.append(0)
            elif "no" in result.lower():
                is_hallucinated.append(1)
            else:
                print(f"Warning: Unexpected response format for sample {i}: {result}")
                is_hallucinated.append(-1)  # Error indicator
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            is_hallucinated.append(-1)  # Error indicator
            results.append("")

    df[f'judge_{model_name}'] = is_hallucinated
    df[f'result_{model_name}'] = results

    return df

def evaluate(df, model_name):
    """Evaluate the model performance"""
    tp, fp, fn = 0, 0, 0
    
    for _, row in df.iterrows():
        if len(row['labels']) == 0:  # no hallucination
            if row[f'judge_{model_name}'] == 1:
                fp += 1
        else: # hallucination
            if row[f'judge_{model_name}'] == 1:
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
    """Main function to run the Groq baseline evaluation"""
    parser = argparse.ArgumentParser(description='Run Groq baseline for hallucination detection')
    parser.add_argument('--data_path', type=str, 
                       default="../../datasets/test/test_w_chunk_score_gpt41mini.json",
                       help='Path to the test data file')
    parser.add_argument('--models', nargs='+', 
                       default=['llama-3.3-70b-versatile', 'openai/gpt-oss-20b', 'qwen/qwen3-32b', 'llama-3.1-8b-instant'],
                       help='List of Groq models to evaluate')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_path', type=str,
                       default='groq_baseline_results.json',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        sys.exit(1)
    
    client = Groq()
    
    # Load and balance data
    print(f"Loading data from: {args.data_path}")
    df = load_and_balance_data(args.data_path)
    
    results = []
    
    # Run evaluation for each model
    for model_name in args.models:
        print(f"\nEvaluating model: {model_name}")
        df = llm_as_a_judge(df, model_name, client)
        result = evaluate(df, model_name)
        results.append(result)
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_path}")
    
    return df, results

if __name__ == "__main__":
    df, results = main()



