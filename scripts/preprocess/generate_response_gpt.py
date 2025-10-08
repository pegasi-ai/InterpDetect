# %%
from transformers import AutoTokenizer
import pandas as pd
import argparse
import sys
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

def load_datasets(train_path, test_path):
    """Load datasets from JSONL files"""
    print("Loading datasets...")
    
    try:
        # Read files and parse JSONL properly with error handling
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        train_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num} in {train_path}: {e}")
                        continue
        
        test_data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        test_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num} in {test_path}: {e}")
                        continue
        
        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)
        
        print(f"Loaded {len(df_train)} training samples, {len(df_test)} test samples")
        
        return df_train, df_test
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

def filter_by_token_count(df_train, df_test, model_name, max_tokens=1024):
    """Filter datasets by token count"""
    print(f"Filtering datasets by token count (max: {max_tokens})...")
    
    # Load tokenizer for counting
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def count_tokens(text: str) -> int:
        encoded = tokenizer(text)
        return len(encoded["input_ids"])
    
    # Count tokens
    df_train["num_tokens"] = df_train["prompt"].apply(count_tokens)
    df_test["num_tokens"] = df_test["prompt"].apply(count_tokens)
    
    # Filter by token count
    df_train = df_train[df_train["num_tokens"] <= max_tokens]
    df_test = df_test[df_test["num_tokens"] <= max_tokens]
    
    print(f"After filtering: {len(df_train)} training, {len(df_test)} test samples")
    
    return df_train, df_test

def limit_samples(df_train, df_test, train_samples, test_samples):
    """Limit the number of samples in each dataset"""
    print(f"Limiting samples: train={train_samples}, test={test_samples}")
    
    df_train = df_train[0:train_samples]
    df_test = df_test[0:test_samples]
    
    print(f"Final dataset sizes: {len(df_train)} training, {len(df_test)} test samples")
    
    return df_train, df_test

def setup_openai_client():
    """Setup OpenAI client"""
    print("Setting up OpenAI client...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Initialize OpenAI client with explicit API key
    try:
        client = OpenAI(api_key=api_key)
    except TypeError as e:
        if "proxies" in str(e):
            # Fallback for httpx compatibility issues
            import httpx
            client = OpenAI(
                api_key=api_key,
                http_client=httpx.Client()
            )
        else:
            raise e
    
    return client

def add_special_template(prompt):
    """Add special template to prompt (for tokenizer compatibility)"""
    # This function is kept for compatibility with the HF version
    # In the GPT version, we don't need to apply chat templates
    return prompt

def generate_response(df, client, model_name, max_new_tokens=1024):
    """Generate responses for the dataset using OpenAI API"""
    print("Generating responses...")
    
    response = []
    response_spans = []
    
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(df)}")
        
        try:
            prompt = row['prompt']
            
            # Create messages for OpenAI API
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Call OpenAI API
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                max_tokens=max_new_tokens,
                temperature=0.3
            )
            
            result = chat_completion.choices[0].message.content
            
            # Parse response into parts and spans
            parts = result.split("\n\n")
            spans = []
            text = ""
            cur = 0
            for part in parts:
                text += part
                spans.append([cur, cur + len(part) - 1])
                cur = cur + len(part)
            
            response.append(text)
            response_spans.append(spans)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            response.append("")
            response_spans.append([])

    df['response'] = response
    df['response_spans'] = response_spans

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
    """Main function to run the response generation pipeline"""
    parser = argparse.ArgumentParser(description='Generate responses for RAGBench dataset using OpenAI GPT models')
    parser.add_argument('--train_path', type=str, 
                       default="datasets/train/train.jsonl",
                       help='Path to training dataset')
    parser.add_argument('--test_path', type=str, 
                       default="datasets/test/test.jsonl",
                       help='Path to test dataset')
    parser.add_argument('--model_name', type=str, 
                       default="gpt-4.1-mini",
                       help='OpenAI model name to use for generation')
    parser.add_argument('--output_dir', type=str, 
                       default="datasets",
                       help='Output directory for generated datasets')
    parser.add_argument('--train_samples', type=int,
                       default=3000,
                       help='Number of training samples to process')
    parser.add_argument('--test_samples', type=int,
                       default=100,
                       help='Number of test samples to process')
    parser.add_argument('--max_tokens', type=int,
                       default=1024,
                       help='Maximum tokens for filtering')
    parser.add_argument('--max_new_tokens', type=int,
                       default=1024,
                       help='Maximum new tokens for generation')
    parser.add_argument('--temperature', type=float,
                       default=0.3,
                       help='Temperature for generation')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip processing training dataset')
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip processing test dataset')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting response generation pipeline...")
    
    # Load datasets
    df_train, df_test = load_datasets(args.train_path, args.test_path)
    
    # Filter by token count
    df_train, df_test = filter_by_token_count(
        df_train, df_test, "Qwen/Qwen3-0.6B", args.max_tokens
    )
    
    # Limit samples
    df_train, df_test = limit_samples(
        df_train, df_test, args.train_samples, args.test_samples
    )
    
    # Setup OpenAI client
    client = setup_openai_client()
    
    # Generate responses for training dataset
    if not args.skip_train:
        print("\nGenerating responses for training dataset...")
        df_train = generate_response(df_train, client, args.model_name, args.max_new_tokens)
        save_dataset(df_train, 
                    os.path.join(args.output_dir, "train", f"train{args.train_samples}_w_response_{args.model_name.replace('-', '')}.jsonl"), 
                    "training")
    
    # Generate responses for test dataset
    if not args.skip_test:
        print("\nGenerating responses for test dataset...")
        df_test = generate_response(df_test, client, args.model_name, args.max_new_tokens)
        save_dataset(df_test, 
                    os.path.join(args.output_dir, "test", f"test{args.test_samples}_w_response_{args.model_name.replace('-', '')}.jsonl"), 
                    "test")
    
    print("Response generation completed successfully!")
    
    # Return processed datasets for potential further use
    return {
        'train': df_train if not args.skip_train else None,
        'test': df_test if not args.skip_test else None
    }

if __name__ == "__main__":
    datasets = main()
