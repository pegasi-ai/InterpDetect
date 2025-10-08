# %%
# !pip install feature_engine
# !pip install xgboost
# !pip install lightgbm
# !pip install optuna
# !pip install --upgrade scikit-learn
# !pip install unidecode

# %%
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import argparse
import sys
import os

def load_data(data_path):
    """Load data from JSON file"""
    print(f"Loading data from {data_path}...")
    
    try:
        with open(data_path, "r") as f:
            response = json.load(f)
        
        print(f"Loaded {len(response)} examples")
        return response
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(response):
    """Preprocess the loaded data into a DataFrame"""
    print("Preprocessing data...")
    
    if not response:
        print("No data to preprocess")
        sys.exit(1)
    
    # Get column names from first example
    ATTENTION_COLS = response[0]['scores'][0]['prompt_attention_score'].keys()
    PARAMETER_COLS = response[0]['scores'][0]['parameter_knowledge_scores'].keys()
    
    data_dict = {
        "identifier": [],
        **{col: [] for col in ATTENTION_COLS},
        **{col: [] for col in PARAMETER_COLS},
        "hallucination_label": []
    }
    
    for i, resp in enumerate(response):
        for j in range(len(resp["scores"])):
            data_dict["identifier"].append(f"response_{i}_item_{j}")
            for col in ATTENTION_COLS:
                data_dict[col].append(resp["scores"][j]['prompt_attention_score'][col])
            
            for col in PARAMETER_COLS:
                data_dict[col].append(resp["scores"][j]['parameter_knowledge_scores'][col])
            data_dict["hallucination_label"].append(resp["scores"][j]["hallucination_label"])
    
    df = pd.DataFrame(data_dict)
    
    print(f"Created DataFrame with {len(df)} samples")
    print(f"Class distribution: {df['hallucination_label'].value_counts().to_dict()}")
    
    return df

def load_model(model_path):
    """Load trained model from pickle file"""
    print(f"Loading model from {model_path}...")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def make_predictions(df, model):
    """Make predictions using the loaded model"""
    print("Making predictions...")
    
    features = [col for col in df.columns if col not in ['identifier', 'hallucination_label']]
    y_pred = model.predict(df[features])
    df['pred'] = y_pred
    
    print(f"Predictions completed for {len(df)} samples")
    return df

def evaluate_span_level(df):
    """Evaluate predictions at span level"""
    print("\n=== Span-level Evaluation ===")
    
    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(df["hallucination_label"], df["pred"]).ravel()
    
    # Precision, recall, F1
    precision = precision_score(df["hallucination_label"], df["pred"])
    recall = recall_score(df["hallucination_label"], df["pred"])
    f1 = f1_score(df["hallucination_label"], df["pred"])
    
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1
    }

def evaluate_response_level(df):
    """Evaluate predictions at response level"""
    print("\n=== Response-level Evaluation ===")
    
    # Extract response_id from identifier (everything before "_item_")
    df["response_id"] = df["identifier"].str.extract(r"(response_\d+)_item_\d+")
    
    # Group by response_id, aggregate with OR (max works for binary 0/1)
    agg_df = df.groupby("response_id").agg({
        "pred": "max",
        "hallucination_label": "max"
    }).reset_index()
    
    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(agg_df["hallucination_label"], agg_df["pred"]).ravel()
    
    # Precision, recall, F1
    precision = precision_score(agg_df["hallucination_label"], agg_df["pred"])
    recall = recall_score(agg_df["hallucination_label"], agg_df["pred"])
    f1 = f1_score(agg_df["hallucination_label"], agg_df["pred"])
    
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'agg_df': agg_df
    }

def save_results(df, span_results, response_results, output_path):
    """Save prediction results and evaluation metrics"""
    print(f"Saving results to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        'span_level': span_results,
        'response_level': response_results,
        'predictions': df.to_dict('records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")

def create_confusion_matrix_plot(df, output_dir, level="span"):
    """Create confusion matrix visualization"""
    print(f"Creating {level}-level confusion matrix plot...")
    
    if level == "response":
        df["response_id"] = df["identifier"].str.extract(r"(response_\d+)_item_\d+")
        agg_df = df.groupby("response_id").agg({
            "pred": "max",
            "hallucination_label": "max"
        }).reset_index()
        y_true = agg_df["hallucination_label"]
        y_pred = agg_df["pred"]
    else:
        y_true = df["hallucination_label"]
        y_pred = df["pred"]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Hallucination', 'Hallucination'],
                yticklabels=['No Hallucination', 'Hallucination'])
    plt.title(f'Confusion Matrix - {level.capitalize()} Level')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(output_dir, f"confusion_matrix_{level}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to {plot_path}")

def main():
    """Main function to run the prediction pipeline"""
    parser = argparse.ArgumentParser(description='Make predictions using trained hallucination detection models')
    parser.add_argument('--data_path', type=str, 
                       default="../datasets/test/test_w_chunk_score_qwen06b.json",
                       help='Path to test data file')
    parser.add_argument('--model_path', type=str, 
                       default="../trained_models/model_SVC_3000.pickle",
                       help='Path to trained model file')
    parser.add_argument('--output_dir', type=str, 
                       default="../results",
                       help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction results to file')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save confusion matrix plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting prediction pipeline...")
    
    # Load data
    response = load_data(args.data_path)
    
    # Preprocess data
    df = preprocess_data(response)
    
    # Load model
    model = load_model(args.model_path)
    
    # Make predictions
    df = make_predictions(df, model)
    
    # Evaluate at span level
    span_results = evaluate_span_level(df)
    
    # Evaluate at response level
    response_results = evaluate_response_level(df)
    
    # Save results if requested
    if args.save_predictions:
        output_path = os.path.join(args.output_dir, "prediction_results.json")
        save_results(df, span_results, response_results, output_path)
    
    # Create plots if requested
    if args.save_plots:
        create_confusion_matrix_plot(df, args.output_dir, "span")
        create_confusion_matrix_plot(df, args.output_dir, "response")
    
    print("\nPrediction pipeline completed successfully!")
    
    return {
        'df': df,
        'span_results': span_results,
        'response_results': response_results
    }

if __name__ == "__main__":
    results = main()


