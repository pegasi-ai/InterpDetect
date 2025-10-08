
#!pip install lettucedetect
# You need to downgrade numpy because of colab
# !pip install numpy==1.26.4
# !pip install torchvision

import pandas as pd
import textwrap
import argparse
import sys
import os
import json

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

def add_labels_llm(df, llama_column, gpt_column):
    """Add binary labels for LLM judge evaluations"""
    print("Adding binary labels for LLM judge evaluations...")
    
    labels_llama = []
    labels_gpt = []

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(df)}")
        
        try:
            # Process Llama labels
            if "Yes" in row[llama_column]:
                labels_llama.append(0)
            elif "No" in row[llama_column]:
                labels_llama.append(1)
            else:
                print(f"Unexpected Llama response format for sample {i}:")
                print(textwrap.fill(row[llama_column], width=120))
                labels_llama.append(-1)  # Error indicator

            # Process GPT labels
            if "Yes" in row[gpt_column]:
                labels_gpt.append(0)
            elif "No" in row[gpt_column]:
                labels_gpt.append(1)
            else:
                print(f"Unexpected GPT response format for sample {i}:")
                print(textwrap.fill(row[gpt_column], width=120))
                labels_gpt.append(-1)  # Error indicator
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            labels_llama.append(-1)
            labels_gpt.append(-1)

    df['labels_llama'] = labels_llama
    df['labels_gpt'] = labels_gpt
    return df

def apply_confidence_threshold(df, threshold=0.7):
    """Apply confidence threshold to LettuceDetect labels (optional)"""
    print(f"Applying confidence threshold: {threshold}")
    
    lst = []

    for _, row in df.iterrows():
        adjusted_labels = []
        for item in row['labels']:
            if item.get('confidence', 1.0) < threshold:
                continue
            adjusted_labels.append(item)

        lst.append(adjusted_labels)

    df['adjusted_labels'] = lst
    return df

def filter_datasets(df_train, df_test, use_confidence_threshold=False, confidence_threshold=0.7):
    """Filter datasets based on LLM judge agreement"""
    print("Filtering datasets based on LLM judge agreement...")
    
    def filtering(df):
        lst = []

        for _, row in df.iterrows():
            if len(row['labels']) == 0:  # no hallucination
                if row['labels_llama'] == 0 or row['labels_gpt'] == 0:
                    lst.append(row)
            else: 
                if row['labels_llama'] == 1 or row['labels_gpt'] == 1:
                    lst.append(row) # hallucination

        return pd.DataFrame(lst)
    
    # Apply confidence threshold if requested
    if use_confidence_threshold:
        df_train = apply_confidence_threshold(df_train, confidence_threshold)
        df_test = apply_confidence_threshold(df_test, confidence_threshold)
    
    # Filter datasets
    df_train_filtered = filtering(df_train)
    df_test_filtered = filtering(df_test)
    
    print(f"After filtering: {len(df_train_filtered)} training, {len(df_test_filtered)} test samples")
    
    return df_train_filtered, df_test_filtered

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
    """Main function to run the filtering pipeline"""
    parser = argparse.ArgumentParser(description='Filter RAGBench datasets based on LLM judge agreement')
    parser.add_argument('--train_path', type=str, 
                       default="../../datasets/train/train3000_w_labels.jsonl",
                       help='Path to training dataset')
    parser.add_argument('--test_path', type=str, 
                       default="../../datasets/test/test1176_w_labels.jsonl",
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, 
                       default="../../tmp",
                       help='Output directory for filtered datasets')
    parser.add_argument('--llama_column', type=str,
                       default="hallucinated_llama-4-maverick-17b-128e-instruct",
                       help='Column name for Llama judge responses')
    parser.add_argument('--gpt_column', type=str,
                       default="hallucinated_gpt-oss-120b",
                       help='Column name for GPT judge responses')
    parser.add_argument('--use_confidence_threshold', action='store_true',
                       help='Apply confidence threshold to LettuceDetect labels')
    parser.add_argument('--confidence_threshold', type=float,
                       default=0.7,
                       help='Confidence threshold for LettuceDetect labels')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip processing training dataset')
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip processing test dataset')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting filtering pipeline...")
    
    # Load datasets
    df_train, df_test = load_datasets(args.train_path, args.test_path)
    
    # Add binary labels for LLM judge evaluations
    print("\nAdding binary labels for LLM judge evaluations...")
    df_train = add_labels_llm(df_train, args.llama_column, args.gpt_column)
    df_test = add_labels_llm(df_test, args.llama_column, args.gpt_column)
    
    # Filter datasets
    print("\nFiltering datasets...")
    df_train_filtered, df_test_filtered = filter_datasets(
        df_train, df_test, args.use_confidence_threshold, args.confidence_threshold
    )
    
    # Save filtered datasets
    if not args.skip_train:
        # Generate output filename from input path
        train_basename = os.path.basename(args.train_path)
        train_output = train_basename.replace("_w_labels.jsonl", "_w_labels_filtered.jsonl")
        save_dataset(df_train_filtered, 
                    os.path.join(args.output_dir, "train", train_output), 
                    "training")
    
    if not args.skip_test:
        # Generate output filename from input path
        test_basename = os.path.basename(args.test_path)
        test_output = test_basename.replace("_w_labels.jsonl", "_w_labels_filtered.jsonl")
        save_dataset(df_test_filtered, 
                    os.path.join(args.output_dir, "test", test_output), 
                    "test")
    
    print("Filtering completed successfully!")
    
    # Return filtered datasets for potential further use
    return {
        'train': df_train_filtered if not args.skip_train else None,
        'test': df_test_filtered if not args.skip_test else None
    }

if __name__ == "__main__":
    datasets = main()



