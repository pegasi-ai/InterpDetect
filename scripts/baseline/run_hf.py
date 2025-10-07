# %%
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
import torch

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

def llm_as_a_judge(df, model_name):
    """Run LLM-as-a-judge evaluation on the dataset using HuggingFace models"""
    print(f"Loading model: {model_name}")
    
    # load the tokenizer and the model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return df

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
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
          #  print(content)

           #print("thinking content:", thinking_content)
            #print("content:", content)

            if "yes" in content.lower():
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
    """Main function to run the HuggingFace baseline evaluation"""
    parser = argparse.ArgumentParser(description='Run HuggingFace baseline for hallucination detection')
    parser.add_argument('--data_path', type=str, 
                       default="../../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json",
                       help='Path to the test data file')
    parser.add_argument('--models', nargs='+', 
                       default=['Qwen/Qwen3-0.6B'],
                       help='List of HuggingFace models to evaluate')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_path', type=str,
                       default='hf_baseline_results.json',
                       help='Path to save results')
    parser.add_argument('--device', type=str,
                       default='auto',
                       help='Device to run models on (auto, cpu, cuda, etc.)')
    
    args = parser.parse_args()
    
    # Set device if specified
    if args.device != 'auto':
        if not torch.cuda.is_available() and args.device == 'cuda':
            print("Warning: CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    # Load and balance data
    print(f"Loading data from: {args.data_path}")
    df = load_and_balance_data(args.data_path)
    
    results = []
    
    # Run evaluation for each model
    for model_name in args.models:
        print(f"\nEvaluating model: {model_name}")
        df = llm_as_a_judge(df, model_name)
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



