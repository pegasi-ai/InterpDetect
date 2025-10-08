# %%
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI
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


# %%
def llm_as_a_judge(df, model_name, client):
    """Run LLM-as-a-judge evaluation on the dataset"""
    is_hallucinated = []

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

            if "yes" in result.lower():
                is_hallucinated.append(0)
            else:
                is_hallucinated.append(1)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            is_hallucinated.append(-1)  # Error indicator

    df[f'judge_{model_name}'] = is_hallucinated

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
    """Main function to run the GPT baseline evaluation"""
    parser = argparse.ArgumentParser(description='Run GPT baseline for hallucination detection')
    parser.add_argument('--data_path', type=str, 
                       default="../../datasets/test/test_w_chunk_score_gpt41mini.json",
                       help='Path to the test data file')
    parser.add_argument('--models', nargs='+', 
                       default=['gpt-5', 'gpt-4.1'],
                       help='List of GPT models to evaluate')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_path', type=str,
                       default='gpt_baseline_results.json',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    client = OpenAI()
    
    # Load and balance data
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    df = load_and_balance_data(data_path)
    
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



