# %%
#!pip install transformer_lens

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import json
from torch.nn import functional as F
from typing import Dict, List, Tuple
import pdb
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import argparse
import sys
import os
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

def load_examples(file_path):
    """Load examples from JSONL file"""
    print(f"Loading examples from {file_path}...")
    
    try:
        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                examples.append(data)
        
        print(f"Loaded {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"Error loading examples: {e}")
        sys.exit(1)

def setup_models(model_name, hf_model_name, device="cuda"):
    """Setup tokenizer, model, and sentence transformer"""
    print(f"Setting up models: {model_name}, {hf_model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        
        model = HookedTransformer.from_pretrained(
            model_name,
            device="cpu",
            torch_dtype=torch.float16
        )
        model.to(device)
        
        bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5").to(device)
        
        return tokenizer, model, bge_model
    except Exception as e:
        print(f"Error setting up models: {e}")
        sys.exit(1)

def calculate_dist_2d(sep_vocabulary_dist, sep_attention_dist):
    """Calculate Jensen-Shannon divergence between distributions"""
    # Calculate softmax
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)

    # Calculate the average distribution M
    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

    # Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)

    # Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').sum(dim=-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').sum(dim=-1)
    js_divs = 0.5 * (kl1 + kl2)

    scores = js_divs.cpu().tolist()
    return sum(scores)

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
    )
    return text

def is_hallucination_span(r_span, hallucination_spans):
    """Check if a span contains hallucination"""
    for token_id in range(r_span[0], r_span[1]):
        for span in hallucination_spans:
            if token_id >= span[0] and token_id <= span[1]:
                return True
    return False

def calculate_hallucination_spans(response, text, response_rag, tokenizer, prefix_len):
    """Calculate hallucination spans"""
    hallucination_span = []
    for item in response:
        start_id = item['start']
        end_id = item['end']
        start_text = text + response_rag[:start_id]
        end_text = text + response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        hallucination_span.append([start_id, end_id])
    return hallucination_span

def calculate_respond_spans(raw_response_spans, text, response_rag, tokenizer):
    """Calculate response spans"""
    respond_spans = []
    for item in raw_response_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = text + response_rag[:start_id]
        end_text = text + response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        respond_spans.append([start_id, end_id])
    return respond_spans

def calculate_prompt_spans(raw_prompt_spans, prompt, tokenizer):
    """Calculate prompt spans"""
    prompt_spans = []
    for item in raw_prompt_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = prompt[:start_id]
        end_text = prompt[:end_id]
        added_start_text = add_special_template(tokenizer, start_text)
        added_end_text = add_special_template(tokenizer, end_text)
        start_text_id = tokenizer(added_start_text, return_tensors="pt").input_ids.shape[-1] - 4
        end_text_id = tokenizer(added_end_text, return_tensors="pt").input_ids.shape[-1] - 4
        prompt_spans.append([start_text_id, end_text_id])
    return prompt_spans

def calculate_sentence_similarity(bge_model, r_text, p_text):
    """Calculate sentence similarity using BGE model"""
    part_embedding = bge_model.encode([r_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([p_text], normalize_embeddings=True)
    
    # Calculate similarity score
    scores_named = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores_named[0])

class MockOutputs:
    """Mock outputs class for transformer lens compatibility"""
    def __init__(self, cache, model_cfg):
        self.cache = cache
        self.model_cfg = model_cfg

    @property
    def attentions(self):
        # Return attention patterns in the expected format
        attentions = []
        for layer in range(self.model_cfg.n_layers):
            # Get attention pattern: [batch, n_heads, seq_len, seq_len]
            attn_pattern = self.cache[f"blocks.{layer}.attn.hook_pattern"]
            attentions.append(attn_pattern)
        return tuple(attentions)

    def __getitem__(self, key):
        if key == "hidden_states":
            # Return hidden states from all layers (residual stream after each layer)
            hidden_states = []
            for layer in range(self.model_cfg.n_layers):
                hidden_state = self.cache[f"blocks.{layer}.hook_resid_post"]
                hidden_states.append(hidden_state)
            return tuple(hidden_states)
        elif key == "logits":
            return logits
        else:
            raise KeyError(f"Key {key} not found")

def process_example(example, tokenizer, model, bge_model, device, max_ctx, iter_step=1):
    """Process a single example to compute scores"""
    response_rag = example['response']
    prompt = example['prompt']
    original_prompt_spans = example['prompt_spans']
    original_response_spans = example['response_spans']

    text = add_special_template(tokenizer, prompt)

    prompt_ids = tokenizer([text], return_tensors="pt").input_ids
    response_ids = tokenizer([response_rag], return_tensors="pt").input_ids
    input_ids = torch.cat([prompt_ids, response_ids[:, 1:]], dim=1)

    if input_ids.shape[-1] > max_ctx:
        overflow = input_ids.shape[-1] - max_ctx
        input_ids = input_ids[:, overflow:]
        prompt_kept = max(prompt_ids.shape[-1] - overflow, 0)
    else:
        prompt_kept = prompt_ids.shape[-1]

    input_ids = input_ids.to(device)
    prefix_len = prompt_kept

    if "labels" in example.keys():
        hallucination_spans = calculate_hallucination_spans(example['labels'], text, response_rag, tokenizer, prefix_len)
    else:
        hallucination_spans = []

    prompt_spans = calculate_prompt_spans(example['prompt_spans'], prompt, tokenizer)
    respond_spans = calculate_respond_spans(example['response_spans'], text, response_rag, tokenizer)

    # Run model with cache to get all intermediate activations
    logits, cache = model.run_with_cache(
        input_ids,
        return_type="logits"
    )

    outputs = MockOutputs(cache, model.cfg)

    # skip tokens without hallucination
    hidden_states = outputs["hidden_states"]
    last_hidden_states = hidden_states[-1][0, :, :]
    del hidden_states

    span_score_dict = []
    for r_id, r_span in enumerate(respond_spans):
        layer_head_span = {}
        parameter_knowledge_dict = {}
        for attentions_layer_id in range(0, model.cfg.n_layers, iter_step):
            for head_id in range(model.cfg.n_heads):
                layer_head = (attentions_layer_id, head_id)
                p_span_score_dict = []
                for p_span in prompt_spans:
                    attention_score = outputs.attentions[attentions_layer_id][0, head_id, :, :]
                    p_span_score_dict.append([p_span, torch.sum(attention_score[r_span[0]:r_span[1], p_span[0]:p_span[1]]).cpu().item()])
                
                # Get the span with maximum score
                p_id = max(range(len(p_span_score_dict)), key=lambda i: p_span_score_dict[i][1])
                prompt_span_text = prompt[original_prompt_spans[p_id][0]:original_prompt_spans[p_id][1]]
                respond_span_text = response_rag[original_response_spans[r_id][0]:original_response_spans[r_id][1]]
                layer_head_span[str(layer_head)] = calculate_sentence_similarity(bge_model, prompt_span_text, respond_span_text)

            x_mid = cache[f"blocks.{attentions_layer_id}.hook_resid_mid"][0, r_span[0]:r_span[1], :]
            x_post = cache[f"blocks.{attentions_layer_id}.hook_resid_post"][0, r_span[0]:r_span[1], :]

            score = calculate_dist_2d(
                x_mid @ model.W_U,
                x_post @ model.W_U
            )
            parameter_knowledge_dict[f"layer_{attentions_layer_id}"] = score

        span_score_dict.append({
            "prompt_attention_score": layer_head_span,
            "r_span": r_span,
            "hallucination_label": 1 if is_hallucination_span(r_span, hallucination_spans) else 0,
            "parameter_knowledge_scores": parameter_knowledge_dict
        })

    example["scores"] = span_score_dict
    return example

def save_batch(select_response, batch_num, save_dir):
    """Save a batch of processed examples"""
    save_path = os.path.join(save_dir, f"train3000_w_chunk_score_part{batch_num}.json")
    with open(save_path, "w") as f:
        json.dump(select_response, f, ensure_ascii=False)
    print(f"Saved batch {batch_num} to {save_path}")

def plot_binary_correlation(numerical_values, binary_labels, title="Correlation with Binary Label"):
    """Plot correlation between numerical values and binary labels"""
    assert len(numerical_values) == len(binary_labels), "Lists must be the same length"

    numerical_values = np.array(numerical_values)
    binary_labels = np.array(binary_labels)

    # Compute correlation
    corr, p_val = pointbiserialr(binary_labels, numerical_values)

    # Plot
    plt.figure(figsize=(8, 3))

    # Scatter plot
    plt.subplot(1, 2, 1)
    sns.stripplot(x=binary_labels, y=numerical_values, jitter=True, alpha=0.7)
    plt.title(f"Scatter Plot\nPoint-Biserial Correlation = {corr:.2f} (p={p_val:.2e})")
    plt.xlabel("Binary Label (0/1)")
    plt.ylabel("Numerical Value")

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=binary_labels, y=numerical_values)
    plt.title("Boxplot by Binary Class")
    plt.xlabel("Binary Label (0/1)")
    plt.ylabel("Numerical Value")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_scores(select_response, save_plots=False, plots_dir="plots"):
    """Analyze computed scores and create visualizations"""
    print("Analyzing scores...")
    
    prompt_attention_scores = []
    hallucination_labels = []
    parameter_knowledge_scores = []
    ratios = []

    for item in select_response:
        scores = item['scores']
        for score in scores:
            pas_sum = sum(score['prompt_attention_score'].values())
            pks_sum = sum(score['parameter_knowledge_scores'].values())
            prompt_attention_scores.append(pas_sum)
            parameter_knowledge_scores.append(pks_sum)
            ratios.append(pks_sum / pas_sum if pas_sum > 0 else 0)
            hallucination_labels.append(score['hallucination_label'])

    # Create plots
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Prompt Attention Scores
        plt.subplot(1, 3, 1)
        plot_binary_correlation(prompt_attention_scores, hallucination_labels, "Correlation with ECS Score")
        
        # Plot 2: Parameter Knowledge Scores
        plt.subplot(1, 3, 2)
        plot_binary_correlation(parameter_knowledge_scores, hallucination_labels, "Correlation with PKS Score")
        
        # Plot 3: Ratio Scores
        plt.subplot(1, 3, 3)
        plot_binary_correlation(ratios, hallucination_labels, "Correlation with Ratio Score")
        
        plt.savefig(os.path.join(plots_dir, "score_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print statistics
    print(f"Score ranges:")
    print(f"Prompt attention scores: {min(prompt_attention_scores):.4f} - {max(prompt_attention_scores):.4f}")
    print(f"Parameter knowledge scores: {min(parameter_knowledge_scores):.4f} - {max(parameter_knowledge_scores):.4f}")
    print(f"Ratios: {min(ratios):.4f} - {max(ratios):.4f}")

def main():
    """Main function to run the score computation pipeline"""
    parser = argparse.ArgumentParser(description='Compute interpretability scores for hallucination detection')
    parser.add_argument('--input_path', type=str, 
                       default="../datasets/train/train3000_w_labels_filtered.jsonl",
                       help='Path to input dataset')
    parser.add_argument('--output_dir', type=str, 
                       default="../datasets/train/chunk_scores",
                       help='Output directory for computed scores')
    parser.add_argument('--model_name', type=str,
                       default="qwen3-0.6b",
                       help='TransformerLens model name')
    parser.add_argument('--hf_model_name', type=str,
                       default="Qwen/Qwen3-0.6B",
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str,
                       default="cuda",
                       help='Device to run models on')
    parser.add_argument('--batch_size', type=int,
                       default=100,
                       help='Batch size for processing')
    parser.add_argument('--iter_step', type=int,
                       default=1,
                       help='Step size for layer iteration')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save analysis plots')
    parser.add_argument('--plots_dir', type=str,
                       default="plots",
                       help='Directory to save plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting score computation pipeline...")
    
    # Load examples
    examples = load_examples(args.input_path)
    
    # Setup models
    tokenizer, model, bge_model = setup_models(args.model_name, args.hf_model_name, args.device)
    
    # Set model to evaluation mode
    model.eval()
    torch.set_grad_enabled(False)
    
    max_ctx = model.cfg.n_ctx
    select_response = []
    
    # Process examples
    for i in tqdm(range(len(examples)), desc="Processing examples"):
        try:
            example = process_example(
                examples[i], tokenizer, model, bge_model, 
                args.device, max_ctx, args.iter_step
            )
            select_response.append(example)
            
            # Save batch if needed
            if (i + 1) % args.batch_size == 0:
                batch_num = i // args.batch_size
                save_batch(select_response, batch_num, args.output_dir)
                select_response = []
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
        
        # Clean up memory
        if i % args.batch_size == 0:
            for name in [
                "input_ids", "logits", "cache", "outputs", "logits_dict",
                "last_hidden_states", "attention_score",
                "parameter_knowledge_scores", "parameter_knowledge_dict",
            ]:
                if name in locals():
                    try:
                        del locals()[name]
                    except Exception:
                        pass
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    
    # Save remaining examples
    if select_response:
        batch_num = len(examples) // args.batch_size
        save_batch(select_response, batch_num, args.output_dir)
    
    # Analyze scores if requested
    if args.save_plots and select_response:
        analyze_scores(select_response, args.save_plots, args.plots_dir)
    
    print("Score computation completed successfully!")
    
    return select_response

if __name__ == "__main__":
    results = main()



