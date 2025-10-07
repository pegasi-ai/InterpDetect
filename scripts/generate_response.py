# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse
import sys
import os
import torch

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

def filter_by_token_count(df_train, df_val, df_test, model_name, max_tokens=1024):
    """Filter datasets by token count"""
    print(f"Filtering datasets by token count (max: {max_tokens})...")
    
    # Load tokenizer for counting
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def count_tokens(text: str) -> int:
        encoded = tokenizer(text)
        return len(encoded["input_ids"])
    
    # Count tokens
    df_train["num_tokens"] = df_train["prompt"].apply(count_tokens)
    df_val["num_tokens"] = df_val["prompt"].apply(count_tokens)
    df_test["num_tokens"] = df_test["prompt"].apply(count_tokens)
    
    # Filter by token count
    df_train = df_train[df_train["num_tokens"] <= max_tokens]
    df_val = df_val[df_val["num_tokens"] <= max_tokens]
    df_test = df_test[df_test["num_tokens"] <= max_tokens]
    
    print(f"After filtering: {len(df_train)} training, {len(df_val)} validation, {len(df_test)} test samples")
    
    return df_train, df_val, df_test

def limit_samples(df_train, df_val, df_test, train_samples, val_samples, test_samples):
    """Limit the number of samples in each dataset"""
    print(f"Limiting samples: train={train_samples}, val={val_samples}, test={test_samples}")
    
    df_train = df_train[0:train_samples]
    df_val = df_val[0:val_samples]
    df_test = df_test[0:test_samples]
    
    print(f"Final dataset sizes: {len(df_train)} training, {len(df_val)} validation, {len(df_test)} test samples")
    
    return df_train, df_val, df_test

def setup_model(model_name, device="auto"):
    """Setup the model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        sys.exit(1)

def add_special_template(tokenizer, prompt):
    """Add special template to prompt"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return text

def generate_response(df, tokenizer, model, max_new_tokens=1024):
    """Generate responses for the dataset"""
    print("Generating responses...")
    
    response = []
    response_spans = []
    
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(df)}")
        
        try:
            prompt = row['prompt']
            text = add_special_template(tokenizer, prompt)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            parts = content.split("\n\n")
            spans = []
            text = ""
            cur = 0
            for part in parts:
                text += part
                spans.append([cur, cur+len(part)-1])
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
    parser = argparse.ArgumentParser(description='Generate responses for RAGBench dataset using language models')
    parser.add_argument('--train_path', type=str, 
                       default="../datasets/train/train.jsonl",
                       help='Path to training dataset')
    parser.add_argument('--val_path', type=str, 
                       default="../datasets/val/val.jsonl",
                       help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, 
                       default="../datasets/test/test.jsonl",
                       help='Path to test dataset')
    parser.add_argument('--model_name', type=str, 
                       default="Qwen/Qwen3-0.6B",
                       help='Model name to use for generation')
    parser.add_argument('--output_dir', type=str, 
                       default="../datasets",
                       help='Output directory for generated datasets')
    parser.add_argument('--train_samples', type=int,
                       default=3000,
                       help='Number of training samples to process')
    parser.add_argument('--val_samples', type=int,
                       default=100,
                       help='Number of validation samples to process')
    parser.add_argument('--test_samples', type=int,
                       default=100,
                       help='Number of test samples to process')
    parser.add_argument('--max_tokens', type=int,
                       default=1024,
                       help='Maximum tokens for filtering')
    parser.add_argument('--max_new_tokens', type=int,
                       default=1024,
                       help='Maximum new tokens for generation')
    parser.add_argument('--device', type=str,
                       default="auto",
                       help='Device to run model on (auto, cpu, cuda, etc.)')
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
        print("Starting response generation pipeline...")
    
    # Load datasets
    df_train, df_val, df_test = load_datasets(args.train_path, args.val_path, args.test_path)
    
    # Filter by token count
    df_train, df_val, df_test = filter_by_token_count(
        df_train, df_val, df_test, args.model_name, args.max_tokens
    )
    
    # Limit samples
    df_train, df_val, df_test = limit_samples(
        df_train, df_val, df_test, args.train_samples, args.val_samples, args.test_samples
    )
    
    # Setup model
    tokenizer, model = setup_model(args.model_name, args.device)
    
    # Generate responses for training dataset
    if not args.skip_train:
        print("\nGenerating responses for training dataset...")
        df_train = generate_response(df_train, tokenizer, model, args.max_new_tokens)
        save_dataset(df_train, 
                    os.path.join(args.output_dir, "train", f"train{args.train_samples}_w_response.jsonl"), 
                    "training")
    
    # Generate responses for validation dataset
    if not args.skip_val:
        print("\nGenerating responses for validation dataset...")
        df_val = generate_response(df_val, tokenizer, model, args.max_new_tokens)
        save_dataset(df_val, 
                    os.path.join(args.output_dir, "val", f"val{args.val_samples}_w_response.jsonl"), 
                    "validation")
    
    # Generate responses for test dataset
    if not args.skip_test:
        print("\nGenerating responses for test dataset...")
        df_test = generate_response(df_test, tokenizer, model, args.max_new_tokens)
        save_dataset(df_test, 
                    os.path.join(args.output_dir, "test", f"test{args.test_samples}_w_response.jsonl"), 
                    "test")
    
    print("Response generation completed successfully!")
    
    # Return processed datasets for potential further use
    return {
        'train': df_train if not args.skip_train else None,
        'val': df_val if not args.skip_val else None,
        'test': df_test if not args.skip_test else None
    }

if __name__ == "__main__":
    datasets = main()



