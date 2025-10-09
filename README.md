# InterpDetect

**InterpDetect: Interpretable Signals for Detecting Hallucinations in Retrieval-Augmented Generation**

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=TZzBKwHLwF)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive framework for detecting and analyzing hallucinations in Retrieval-Augmented Generation (RAG) systems using interpretability techniques and chunk-level analysis on the RAGBench/FinQA dataset.

## Overview

This project implements a novel approach to hallucination detection by leveraging interpretability methods. The framework consists of three main parts:

1. **Preprocessing Pipeline** - Convert raw datasets to labeled data with hallucination spans
2. **Training & Prediction** - Train classifiers or use pre-trained models for hallucination detection
3. **Baseline Comparisons** - Evaluate against multiple baseline methods (RAGAS, TruLens, RefChecker, GPT-based, etc.)

The framework can work with existing labeled datasets or process raw data through the complete pipeline.

## Features

- **Interpretability Focus**: Uses interpretability techniques to understand model decisions
- **Chunk-level Analysis**: Breaks down responses into chunks and analyzes each for hallucination likelihood
- **Trained Models**: Pre-trained machine learning models (Logistic Regression, Random Forest, SVC, XGBoost) for hallucination detection
- **Multiple Baselines**: Implements various baseline methods including GPT, Groq, HuggingFace models, RAGAS, RefChecker, and TruLens


## Project Structure

```
interpretablity-hallucination-detection/
├── datasets/                    # Data files
│   ├── OV_copying_score.json   # Overlap copying scores
│   ├── test/                   # Chunk-level scores for testing
│   └── train/                  # Chunk-level scores for training
├── scripts/                    # Python scripts for pipeline execution
|   └── baseline/               # Baseline implementations
|       |── requirements.txt    # Python dependencies for baselines
│       ├── run_gpt.py          # GPT baseline
│       ├── run_groq.py         # Groq baseline
│       ├── run_hf.py           # HuggingFace baseline
│       ├── run_ragas.py        # RAGAS baseline
│       ├── run_refchecker.py   # RefChecker baseline
│       └── run_trulens.py      # TruLens baseline
│   ├── preprocess              # Data preprocessing
|       |── datasets            # Preprocessed Train and Test
|       |── preprocess.py           # 1. add prompt and prompt_spans to raw data
│       ├── generate_response.py    # 2. Response generation (either hf models or gpt)
│       ├── generate_labels.py      # 3. Generate Hallucination labels and Add LLM-as-a-Judge
│       ├── filter.py               # 4. Run majority voting to filter out low confident prediction
|       ├── helper.py               # Utility functions
│   ├── compute_scores.py       # Chunk-level score computation
│   ├── classifier.py           # Model training 
│   ├── predict.py              # Model prediction    
│   
├── trained_models/             # Pre-trained ML models
│   ├── model_LR_3000.pickle    # Logistic Regression model
│   ├── model_RandomForest_3000.pickle # Random Forest model
│   ├── model_SVC_3000.pickle   # Support Vector Classifier
│   └── model_XGBoost_3000.pickle # XGBoost model
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd InterpDetect
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (create a `.env` file):
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key

# Optional: HuggingFace token for private datasets
HUGGINGFACE_TOKEN=your_hf_token
```

## Usage

The framework consists of three main parts that can be used independently or together:

### Part 1: Preprocessing Pipeline (Optional)

**Skip this part if you already have datasets with the required format:**
- Required columns: `prompt`, `prompt_spans`, `response`, `response_spans`, `labels` (containing hallucinated spans)

If you need to process raw data, see the [Preprocessing README](scripts/preprocess/README.md) for detailed instructions.

**Quick preprocessing workflow:**

```bash
# Step 1: Generate responses using GPT
python scripts/preprocess/generate_response_gpt.py \
  --model_name "gpt-4.1-mini" \
  --train_samples 3000 \
  --test_samples 1176

# Step 2: Generate hallucination labels
python scripts/preprocess/generate_labels.py \
  --llm_client "groq" \
  --llm_model "llama-3.1-70b-versatile"

# Step 3: Filter datasets based on confidence
python scripts/preprocess/filter.py \
  --use_confidence_threshold \
  --confidence_threshold 0.8
```

See [scripts/preprocess/README.md](scripts/preprocess/README.md) for complete documentation.

---

### Part 2: Training & Prediction

This part computes interpretability scores (PKS and ECS) and trains/uses classifiers for hallucination detection.

#### Option A: Direct Prediction (Using Pre-trained Models)

Use this if you want to predict without training:

```bash
# Step 1: Compute PKS and ECS scores for test data
python scripts/compute_scores.py \
    --input_path "datasets/test/test1176_w_labels_filtered.jsonl" \
    --output_dir "datasets/test" \
    --model_name "Qwen/Qwen3-0.6B" \
    --device "cpu"

# Step 2: Run prediction using a pre-trained model
python scripts/predict.py \
    --data_path "datasets/test/test1176_w_chunk_score.json" \
    --model_path "trained_models/model_XGBoost_3000.pickle" \
    --output_dir "results" \
    --save_predictions \
    --save_plots
```

#### Option B: Train Your Own Classifier

Use this to train a new classifier on your data:

```bash
# Step 1: Compute scores for both training and test data
python scripts/compute_scores.py \
    --input_path "datasets/train/train3000_w_labels_filtered.jsonl" \
    --output_dir "datasets/train/chunk_scores" \
    --model_name "Qwen/Qwen3-0.6B"

python scripts/compute_scores.py \
    --input_path "datasets/test/test1176_w_labels_filtered.jsonl" \
    --output_dir "datasets/test" \
    --model_name "Qwen/Qwen3-0.6B"

python scripts/classifier.py \
    --input_dir "datasets/train/chunk_scores" \
    --output_dir "trained_models" \
    --models "LogisticRegression" "RandomForest" "SVC" "XGBoost" \
    --test_size 0.2 \
    --balance_classes

# Step 3: Run prediction with your trained model
python scripts/predict.py \
    --data_path "datasets/test/test1176_w_chunk_score.json" \
    --model_path "trained_models/model_XGBoost_3000.pickle" \
    --output_dir "results" \
    --save_predictions \
    --save_plots
```

**Available Pre-trained Models:**
- `model_LR_3000.pickle` - Logistic Regression
- `model_RandomForest_3000.pickle` - Random Forest
- `model_SVC_3000.pickle` - Support Vector Classifier
- `model_XGBoost_3000.pickle` - XGBoost (recommended)

---

### Part 3: Baseline Comparisons

Run various baseline methods to compare against your approach. Baselines require additional dependencies (see `scripts/baseline/requirements.txt`).

**Available Baselines:**

```bash
# GPT Baseline
python scripts/baseline/run_gpt.py \
    --data_path "datasets/test/test1176_w_chunk_score.json" \
    --models "gpt-4o-mini"

# Groq Baseline (Llama models)
python scripts/baseline/run_groq.py \
    --data_path "datasets/test/test1176_w_chunk_score.json" \
    --models "llama3-70b-8192"

# HuggingFace Models Baseline
python scripts/baseline/run_hf.py \
    --data_path "datasets/test/test1176_w_chunk_score.json" \
    --models "Qwen/Qwen3-0.6B"

# RAGAS Baseline
python scripts/baseline/run_ragas.py \
    --data_path "datasets/test/test1176_w_chunk_score.json"

# RefChecker Baseline
python scripts/baseline/run_refchecker.py \
    --data_path "datasets/test/test1176_w_chunk_score.json"

# TruLens Baseline
python scripts/baseline/run_trulens.py \
    --data_path "datasets/test/test1176_w_chunk_score.json"
```

**Install baseline dependencies:**
```bash
pip install -r scripts/baseline/requirements.txt
```

---

## Data Format

### Required Dataset Format

For Parts 2 and 3, your dataset must include:

**Required columns:**
- `prompt`: The input question/prompt
- `prompt_spans`: Span information for the prompt
- `response`: The model's generated response
- `response_spans`: Span information for the response  
- `labels`: List of hallucinated spans in the response

**Example:**
```json
{
  "id": "finqa_123",
  "question": "What is the revenue?",
  "documents": ["Company revenue was $100M..."],
  "prompt": "Given the context...",
  "prompt_spans": [[0, 150]],
  "response": "The revenue is $100M",
  "response_spans": [[0, 20]],
  "labels": []
}
```

### Output Files

**After compute_scores.py:**
- Chunk-level PKS (Parameter Knowledge Score) and ECS (Embedding Cosine Similarity) for each response chunk
- JSON format with scores per chunk

**After classifier.py:**
- Trained model files (`.pickle` format)
- Training metrics and plots

**After predict.py:**
- Predictions with confidence scores
- Evaluation metrics (precision, recall, F1-score, AUC-ROC)
- Confusion matrix and performance plots

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional
HUGGINGFACE_TOKEN=your_hf_token_here
```

## Citation

If you use this code or our work in your research, please cite our paper:

```bibtex
@inproceedings{tan2025interpdetect,
  title={InterpDetect: Interpretable Signals for Detecting Hallucinations in Retrieval-Augmented Generation},
  author={Tan, Likun and Huang, Kuan-Wei and Shi, Joy and Wu, Kevin},
  booktitle={OpenReview},
  year={2025},
  url={https://openreview.net/pdf?id=TZzBKwHLwF}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the open-source community for the various baseline implementations
- Contributors to the interpretability research community, especially the TransformerLens team
- The RAGBench team for providing the FinQA dataset
- Users and testers of this framework