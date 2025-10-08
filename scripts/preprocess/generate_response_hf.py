# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse
import sys
import os
import json
import torch

def load_datasets(train_path, test_path):
    """Load datasets from JSONL files"""
    print("Loading datasets...")
    
    try:
        # Use StringIO to avoid FutureWarning about literal json
        from io import StringIO
        
        # Read files and parse JSONL properly
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line.strip()) for line in f if line.strip()]
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line.strip()) for line in f if line.strip()]
        
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
    
    print(f"After filtering: {len(df_train)} training,  {len(df_test)} test samples")
    
    return df_train, df_test

def limit_samples(df_train, df_test, train_samples, test_samples):
    """Limit the number of samples in each dataset"""
    print(f"Limiting samples: train={train_samples}, test={test_samples}")
    
    df_train = df_train[0:train_samples]
    df_test = df_test[0:test_samples]
    
    print(f"Final dataset sizes: {len(df_train)} training, {len(df_test)} test samples")
    
    return df_train, df_test

def setup_model(model_name, device="auto"):
    """Setup the model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    # Handle device selection for Apple Silicon
    if device == "auto":
        if torch.backends.mps.is_available():
            # For certain models, MPS can cause matrix multiplication issues
            # Use CPU for stability unless explicitly requested
            print("MPS available but using CPU for stability (use --device mps to force MPS)")
            device = "cpu"
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA acceleration")
        else:
            device = "cpu"
            print("Using CPU")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings for Apple Silicon
        if device == "mps":
            # Use float32 for MPS to avoid matrix multiplication issues
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"  # Load to CPU first, then move to MPS
            )
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device
            )
        
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model with {device}, falling back to CPU: {e}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            return tokenizer, model, "cpu"
        except Exception as e2:
            print(f"Error loading model {model_name}: {e2}")
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

def generate_response(df, tokenizer, model, max_new_tokens=1024, device=None):
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
            model_inputs = tokenizer([text], return_tensors="pt")
            
            # Move inputs to the correct device
            if device:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            else:
                model_inputs = model_inputs.to(model.device)
            
            # conduct text completion with MPS error handling
            try:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens
                )
            except RuntimeError as e:
                if "mps_matmul" in str(e) or "incompatible dimensions" in str(e):
                    print(f"MPS error detected, falling back to CPU for sample {i}")
                    # Move model and inputs to CPU
                    model = model.cpu()
                    model_inputs = {k: v.cpu() for k, v in model_inputs.items()}
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens
                    )
                else:
                    raise e
            output_ids = generated_ids[0][len(model_inputs['input_ids'][0]):].tolist() 

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
                       default="datasets/train/train.jsonl",
                       help='Path to training dataset')
    parser.add_argument('--test_path', type=str, 
                       default="datasets/test/test.jsonl",
                       help='Path to test dataset')
    parser.add_argument('--model_name', type=str, 
                       default="Qwen/Qwen3-0.6B",
                       help='Model name to use for generation')
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
    parser.add_argument('--device', type=str,
                       default="auto",
                       help='Device to run model on (auto, cpu, cuda, etc.)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage to avoid MPS issues on Apple Silicon')
    parser.add_argument('--disable_mps', action='store_true',
                       help='Disable MPS acceleration (recommended for Apple Silicon)')
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
        df_train, df_test, args.model_name, args.max_tokens
    )
    
    # Limit samples
    df_train, df_test = limit_samples(
        df_train, df_test, args.train_samples, args.test_samples
    )
    
    # Setup model
    if args.force_cpu or args.disable_mps:
        device = "cpu"
    else:
        device = args.device
    tokenizer, model, device = setup_model(args.model_name, device)
    
    # Generate responses for training dataset
    if not args.skip_train:
        print("\nGenerating responses for training dataset...")
        df_train = generate_response(df_train, tokenizer, model, args.max_new_tokens, device)
        save_dataset(df_train, 
                    os.path.join(args.output_dir, "train", f"train{args.train_samples}_w_response.jsonl"), 
                    "training")
    
    # Generate responses for test dataset
    if not args.skip_test:
        print("\nGenerating responses for test dataset...")
        df_test = generate_response(df_test, tokenizer, model, args.max_new_tokens, device)
        save_dataset(df_test, 
                    os.path.join(args.output_dir, "test", f"test{args.test_samples}_w_response.jsonl"), 
                    "test")
    
    print("Response generation completed successfully!")
    
    # Return processed datasets for potential further use
    return {
        'train': df_train if not args.skip_train else None,
        'test': df_test if not args.skip_test else None
    }

if __name__ == "__main__":
    datasets = main()



