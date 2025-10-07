# %%
from datasets import load_dataset
import pandas as pd
from helper import clean_text
import argparse
import sys
import os

def load_data_from_hf():
    """Load data from HuggingFace datasets"""
    print("Loading data from HuggingFace...")
    
    # Login using e.g. `huggingface-cli login` to access this dataset
    splits = {
        'train': 'finqa/train-00000-of-00001.parquet', 
        'test': 'finqa/test-00000-of-00001.parquet', 
        'validation': 'finqa/validation-00000-of-00001.parquet'
    }
    
    try:
        df_train = pd.read_parquet("hf://datasets/galileo-ai/ragbench/" + splits["train"])
        df_val = pd.read_parquet("hf://datasets/galileo-ai/ragbench/" + splits["validation"])
        df_test = pd.read_parquet("hf://datasets/galileo-ai/ragbench/" + splits["test"])
        
        print(f"Loaded {len(df_train)} training samples, {len(df_val)} validation samples, {len(df_test)} test samples")
        
        return df_train, df_val, df_test
        
    except Exception as e:
        print(f"Error loading data from HuggingFace: {e}")
        print("Please make sure you are logged in with `huggingface-cli login`")
        sys.exit(1)

def add_prompt_spans(df):
    """Build prompt and compute spans for the dataset"""
    part1 = "Given the context, please answer the question based on the provided information from the context. Include any reasoning with the answer\n"
    part2 = "\nContext:"
    part3 = "\nQuestion:"
    part4 = "\nAnswer:"

    prompt_texts = []
    prompt_spans = []

    for i, row in df.iterrows():
        question = row["question"]
        docs = list(row["documents"])  # assume list of document strings
        
        # prefix
        prompt = ""
        spans = []
        l1 = len(part1)
        prompt+=part1
        spans.append([0, l1-1])
        
        # context
        l2 = len(part2)
        prompt+=part2
        spans.append([l1, l1+l2-1])
        cur = l1+l2
        for doc in docs:
            doc = clean_text(doc)
            prompt+=doc
            spans.append([cur, cur+len(doc)-1])
            cur = cur+len(doc)

        # question
        l3 = len(part3)
        prompt+=part3
        spans.append([cur, cur+l3-1])
        cur = cur+l3
        prompt+=question
        spans.append([cur, cur+len(question)-1])
        cur = cur+len(question)
        
        # answer
        l4 = len(part4)
        prompt+=part4
        spans.append([cur, cur+l4-1])

        # append
        prompt_texts.append(prompt)
        prompt_spans.append(spans)

    return prompt_texts, prompt_spans

def process_dataset(df, dataset_name):
    """Process a single dataset by adding prompts and spans"""
    print(f"Processing {dataset_name} dataset...")
    
    prompts, spans = add_prompt_spans(df)
    df['prompt'] = prompts
    df['prompt_spans'] = spans
    
    # Select required columns
    COLS = ['id', 'question', 'documents', 'documents_sentences', 'prompt', 'prompt_spans']
    df = df[COLS]
    
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
    """Main function to run the preprocessing pipeline"""
    parser = argparse.ArgumentParser(description='Preprocess RAGBench dataset for hallucination detection')
    parser.add_argument('--output_dir', type=str, 
                       default="../datasets",
                       help='Output directory for processed datasets')
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
        print("Starting preprocessing pipeline...")
    
    # Load data from HuggingFace
    df_train, df_val, df_test = load_data_from_hf()
    
    # Process and save training dataset
    if not args.skip_train:
        df_train_processed = process_dataset(df_train, "training")
        save_dataset(df_train_processed, 
                    os.path.join(args.output_dir, "train", "train.jsonl"), 
                    "training")
    
    # Process and save validation dataset
    if not args.skip_val:
        df_val_processed = process_dataset(df_val, "validation")
        save_dataset(df_val_processed, 
                    os.path.join(args.output_dir, "val", "val.jsonl"), 
                    "validation")
    
    # Process and save test dataset
    if not args.skip_test:
        df_test_processed = process_dataset(df_test, "test")
        save_dataset(df_test_processed, 
                    os.path.join(args.output_dir, "test", "test.jsonl"), 
                    "test")
    
    print("Preprocessing completed successfully!")
    
    # Return processed datasets for potential further use
    return {
        'train': df_train_processed if not args.skip_train else None,
        'val': df_val_processed if not args.skip_val else None,
        'test': df_test_processed if not args.skip_test else None
    }

if __name__ == "__main__":
    datasets = main()



