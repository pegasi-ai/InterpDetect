#!pip install lettucedetect
# You need to downgrade numpy because of colab
# !pip install numpy==1.26.4
# !pip install torchvision

import pandas as pd
import textwrap
import argparse
import sys
import os
from dotenv import load_dotenv

def load_datasets(train_path, val_path, test_path):
    """Load datasets from JSONL files"""
    print("Loading datasets...")
    
    try:
        df_train = pd.read_json(train_path, lines=True)
        df_val = pd.read_json(val_path, lines=True)
        df_test = pd.read_json(test_path, lines=True)
        
        print(f"Loaded {len(df_train)} training samples, {len(df_val)} validation samples, {len(df_test)} test samples")
        
        return df_train, df_val, df_test
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

def setup_lettuce_detector(method="transformer", model_path="KRLabsOrg/lettucedect-large-modernbert-en-v1"):
    """Setup LettuceDetect hallucination detector"""
    print(f"Setting up LettuceDetect with method: {method}, model: {model_path}")
    
    try:
        from lettucedetect.models.inference import HallucinationDetector
        
        detector = HallucinationDetector(
            method=method, model_path=model_path
        )
        return detector
    except Exception as e:
        print(f"Error setting up LettuceDetect: {e}")
        sys.exit(1)

def add_lettuce_labels(df, detector):
    """Add span-level labels using LettuceDetect"""
    print("Adding LettuceDetect labels...")
    
    labels = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(df)}")
        
        try:
            contexts = row["documents"]
            question = row['question']
            answer = row['response']

            # Get span-level predictions indicating which parts of the answer are considered hallucinated.
            predictions = detector.predict(
                context=contexts, question=question, answer=answer, output_format="spans"
            )
            label = []
            for pred in predictions:
                label.append(pred)

            labels.append(label)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            labels.append([])

    df['labels'] = labels
    return df

def setup_llm_client(client_type="groq"):
    """Setup LLM client for judge evaluation"""
    print(f"Setting up {client_type} client...")
    
    load_dotenv()
    
    if client_type.lower() == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables")
            sys.exit(1)
        return OpenAI()
    elif client_type.lower() == "groq":
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Error: GROQ_API_KEY not found in environment variables")
            sys.exit(1)
        return Groq()
    else:
        print(f"Unsupported client type: {client_type}")
        sys.exit(1)

def generate_judge_prompt(context: str, question: str, response: str) -> str:
    """Generate prompt for LLM-as-a-judge evaluation"""
    prompt = f"""
    You are an expert fact-checker. Given a context, a question, and a response, determine if the response is faithful to the context.

    Context:
    {context}

    Question:
    {question}

    Response:
    {response}

    Output format:
    1. "Yes" if the response is fully supported by the context.
    2. "No" if any part is unsupported, followed by a concise list of unsupported parts.
    Be objective and concise.
    """
    return textwrap.dedent(prompt).strip()

def add_llm_judge(df, client, model):
    """Add LLM-as-a-judge labels"""
    print(f"Adding LLM-as-a-judge labels with model: {model}")
    
    hallucinated_llm = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(df)}")

        try:
            docs = list(row['documents'])
            doc = " ".join(docs)
            question = row['question']
            response = row['response']
            prompt = generate_judge_prompt(doc, question, response)
            messages = [{"role": "user", "content": prompt}]

            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
            )

            result = chat_completion.choices[0].message.content
            hallucinated_llm.append(result)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            hallucinated_llm.append("")
    
    model_name = model.split("/")[-1]
    df[f'hallucinated_{model_name}'] = hallucinated_llm

    return df

def save_dataset(df, output_path, dataset_name):
    """Save dataset to JSONL format"""
    print(f"Saving {dataset_name} dataset to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        print(f"Successfully saved {len(df)} samples to {output_path}")
    except Exception as e:
        print(f"Error saving {dataset_name} dataset: {e}")
        sys.exit(1)

def main():
    """Main function to run the label generation pipeline"""
    parser = argparse.ArgumentParser(description='Generate labels for RAGBench dataset using LettuceDetect and LLM-as-a-judge')
    parser.add_argument('--train_path', type=str, 
                       default="../datasets/train/train3000_w_response.jsonl",
                       help='Path to training dataset')
    parser.add_argument('--val_path', type=str, 
                       default="../datasets/val/val100_w_response.jsonl",
                       help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, 
                       default="../datasets/test/test100_w_response.jsonl",
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, 
                       default="../datasets",
                       help='Output directory for labeled datasets')
    parser.add_argument('--lettuce_method', type=str,
                       default="transformer",
                       help='LettuceDetect method to use')
    parser.add_argument('--lettuce_model', type=str,
                       default="KRLabsOrg/lettucedect-large-modernbert-en-v1",
                       help='LettuceDetect model path')
    parser.add_argument('--llm_client', type=str,
                       default="groq",
                       choices=["openai", "groq"],
                       help='LLM client to use for judge evaluation')
    parser.add_argument('--llm_models', nargs='+',
                       default=["meta-llama/llama-4-maverick-17b-128e-instruct", "openai/gpt-oss-120b"],
                       help='LLM models to use for judge evaluation')
    parser.add_argument('--skip_lettuce', action='store_true',
                       help='Skip LettuceDetect labeling')
    parser.add_argument('--skip_llm_judge', action='store_true',
                       help='Skip LLM-as-a-judge labeling')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip processing training dataset')
    parser.add_argument('--skip_val', action='store_true',
                       help='Skip processing validation dataset')
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip processing test dataset')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting label generation pipeline...")
    
    # Load datasets
    df_train, df_val, df_test = load_datasets(args.train_path, args.val_path, args.test_path)
    
    # Setup LettuceDetect detector
    if not args.skip_lettuce:
        detector = setup_lettuce_detector(args.lettuce_method, args.lettuce_model)
    
    # Setup LLM client
    if not args.skip_llm_judge:
        client = setup_llm_client(args.llm_client)
    
    # Process training dataset
    if not args.skip_train:
        print("\nProcessing training dataset...")
        if not args.skip_lettuce:
            df_train = add_lettuce_labels(df_train, detector)
        if not args.skip_llm_judge:
            for model in args.llm_models:
                df_train = add_llm_judge(df_train, client, model)
        save_dataset(df_train, 
                    os.path.join(args.output_dir, "train", "train3000_w_labels.jsonl"), 
                    "training")
    
    # Process validation dataset
    if not args.skip_val:
        print("\nProcessing validation dataset...")
        if not args.skip_lettuce:
            df_val = add_lettuce_labels(df_val, detector)
        if not args.skip_llm_judge:
            for model in args.llm_models:
                df_val = add_llm_judge(df_val, client, model)
        save_dataset(df_val, 
                    os.path.join(args.output_dir, "val", "val100_w_labels.jsonl"), 
                    "validation")
    
    # Process test dataset
    if not args.skip_test:
        print("\nProcessing test dataset...")
        if not args.skip_lettuce:
            df_test = add_lettuce_labels(df_test, detector)
        if not args.skip_llm_judge:
            for model in args.llm_models:
                df_test = add_llm_judge(df_test, client, model)
        save_dataset(df_test, 
                    os.path.join(args.output_dir, "test", "test100_w_labels.jsonl"), 
                    "test")
    
    print("Label generation completed successfully!")
    
    # Return processed datasets for potential further use
    return {
        'train': df_train if not args.skip_train else None,
        'val': df_val if not args.skip_val else None,
        'test': df_test if not args.skip_test else None
    }

if __name__ == "__main__":
    datasets = main()



