# %%
# !pip install feature_engine
# !pip install xgboost
# !pip install lightgbm
# !pip install optuna
# !pip install --upgrade scikit-learn
# !pip install unidecode

import pandas as pd
import json
import numpy as np
import os
import glob
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import argparse
import sys

def load_data(folder_path):
    """Load data from JSON files in the specified folder"""
    print(f"Loading data from {folder_path}...")
    
    try:
        response = []
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {folder_path}")
            sys.exit(1)
        
        for file_path in json_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                response.extend(data)
        
        print(f"Loaded {len(response)} examples from {len(json_files)} files")
        return response
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(response, balance_classes=True, random_state=42):
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
    
    # Balance classes if requested
    if balance_classes:
        min_count = df['hallucination_label'].value_counts().min()
        df = (
            df.groupby('hallucination_label', group_keys=False)
              .apply(lambda x: x.sample(min_count, random_state=random_state))
        )
        print(f"After balancing: {df['hallucination_label'].value_counts().to_dict()}")
    
    return df, list(ATTENTION_COLS), list(PARAMETER_COLS)

def split_data(df, test_size=0.1, random_state=42):
    """Split data into train and validation sets"""
    print("Splitting data into train and validation sets...")
    
    train, val = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['hallucination_label'])
    
    features = [col for col in df.columns if col not in ['identifier', 'hallucination_label']]
    
    X_train = train[features]
    y_train = train["hallucination_label"]
    X_val = val[features]
    y_val = val["hallucination_label"]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Number of features: {len(features)}")
    
    return X_train, X_val, y_train, y_val, features

def create_preprocessor(use_feature_selection=False):
    """Create preprocessing pipeline"""
    from sklearn.preprocessing import StandardScaler
    from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection, DropDuplicateFeatures
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    scaler = StandardScaler()
    
    if use_feature_selection:
        drop_const = DropConstantFeatures(tol=0.95, missing_values='ignore')
        drop_dup = DropDuplicateFeatures()
        drop_corr = SmartCorrelatedSelection(
            method='pearson', 
            threshold=0.90,
            selection_method='model_performance',
            estimator=RandomForestClassifier(max_depth=5, random_state=42)
        )
        
        preprocessor = Pipeline([
            ('scaler', scaler),
            ('drop_constant', drop_const),
            ('drop_duplicates', drop_dup),
            ('smart_corr_selection', drop_corr),
        ])
    else:
        preprocessor = Pipeline([
            ('scaler', scaler),
        ])
    
    return preprocessor

def train_models(X_train, X_val, y_train, y_val, preprocessor, models_to_train=None):
    """Train multiple models and compare their performance"""
    print("Training models...")
    
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    
    # Define models to train
    if models_to_train is None:
        models_to_train = ["LR", "SVC", "RandomForest", "XGBoost"]
    
    models = []
    if "LR" in models_to_train:
        models.append(("LR", LogisticRegression()))
    if "SVC" in models_to_train:
        models.append(('SVC', SVC()))
    if "RandomForest" in models_to_train:
        models.append(('RandomForest', RandomForestClassifier(max_depth=5)))
    if "XGBoost" in models_to_train:
        models.append(('XGBoost', XGBClassifier(max_depth=5)))
    
    # Initialize lists for results
    names = []
    train_ps = []
    train_rs = []
    train_fs = []
    val_ps = []
    val_rs = []
    val_fs = []
    clfs = {}
    
    # Train each model
    for name, model in models:
        print(f"Training {name}...")
        names.append(name)
        clf = make_pipeline(preprocessor, model)
        clf.fit(X_train, y_train)
        
        # Calculate metrics
        tp, tr, tf, _ = precision_recall_fscore_support(y_train, clf.predict(X_train), average='binary')
        train_ps.append(tp)
        train_rs.append(tr)
        train_fs.append(tf)
        
        vp, vr, vf, _ = precision_recall_fscore_support(y_val, clf.predict(X_val), average='binary')
        val_ps.append(vp)
        val_rs.append(vr)
        val_fs.append(vf)
        
        clfs[name] = clf
    
    # Create comparison dataframe
    model_comparison = pd.DataFrame({
        'Algorithm': names,
        'Train_p': train_ps,
        'Val_p': val_ps,
        'Train_r': train_rs,
        'Val_r': val_rs,
        'Train_f': train_fs,
        'Val_f': val_fs,
    })
    
    print("\nModel Comparison:")
    print(model_comparison)
    
    return clfs, model_comparison

def save_models(clfs, output_dir):
    """Save trained models"""
    print(f"Saving models to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, clf in clfs.items():
        model_path = os.path.join(output_dir, f"model_{name}_3000.pickle")
        with open(model_path, "wb") as fout:
            pickle.dump(clf, fout)
        print(f"Saved {name} model to {model_path}")



def create_feature_importance_plot(clfs, X_train, output_dir):
    """Create feature importance plot for XGBoost model"""
    print("Creating feature importance plot...")
    
    if 'XGBoost' in clfs:
        xgb_model = clfs['XGBoost']
        feature_imp = pd.DataFrame(
            sorted(zip(xgb_model.named_steps['xgbclassifier'].feature_importances_, X_train.columns)), 
            columns=['Value', 'Feature']
        )
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:15])
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved feature importance plot to {plot_path}")

def main():
    """Main function to run the classifier training pipeline"""
    parser = argparse.ArgumentParser(description='Train classifiers for hallucination detection')
    parser.add_argument('--input_dir', type=str, 
                       default="../datasets/train/chunk_scores",
                       help='Input directory containing JSON files with scores')
    parser.add_argument('--output_dir', type=str, 
                       default="../trained_models",
                       help='Output directory for trained models')
    parser.add_argument('--models', nargs='+',
                       default=["LR", "SVC", "RandomForest", "XGBoost"],
                       choices=["LR", "SVC", "RandomForest", "XGBoost"],
                       help='Models to train')
    parser.add_argument('--test_size', type=float,
                       default=0.1,
                       help='Test size for train/validation split')
    parser.add_argument('--balance_classes', action='store_true',
                       help='Balance classes by undersampling')
    parser.add_argument('--use_feature_selection', action='store_true',
                       help='Use feature selection in preprocessing')

    parser.add_argument('--random_state', type=int,
                       default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save feature importance plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting classifier training pipeline...")
    
    # Load data
    response = load_data(args.input_dir)
    
    # Preprocess data
    df, attention_cols, parameter_cols = preprocess_data(
        response, args.balance_classes, args.random_state
    )
    
    # Split data
    X_train, X_val, y_train, y_val, features = split_data(
        df, args.test_size, args.random_state
    )
    
    # Create preprocessor
    preprocessor = create_preprocessor(args.use_feature_selection)
    
    # Train models
    clfs, model_comparison = train_models(
        X_train, X_val, y_train, y_val, preprocessor, args.models
    )
    
    # Save models
    save_models(clfs, args.output_dir)
    
    # Create feature importance plot if requested
    if args.save_plots:
        create_feature_importance_plot(clfs, X_train, args.output_dir)
    
    print("Classifier training completed successfully!")
    
    return {
        'models': clfs,
        'model_comparison': model_comparison,
        'features': features
    }

if __name__ == "__main__":
    results = main()



