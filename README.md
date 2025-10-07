# InterpDetect

A comprehensive framework for detecting and analyzing hallucinations in language models using interpretability techniques and chunk-level analysis.

## Overview

This project implements a novel approach to hallucination detection by leveraging interpretability methods and chunk-level scoring. The framework includes preprocessing pipelines, multiple baseline implementations, and trained models for detecting hallucinations in language model responses.

## Features

- **Chunk-level Analysis**: Breaks down responses into chunks and analyzes each for hallucination likelihood
- **Multiple Baselines**: Implements various baseline methods including GPT, Groq, HuggingFace models, RAGAS, RefChecker, and TruLens
- **Trained Models**: Pre-trained machine learning models (Logistic Regression, Random Forest, SVC, XGBoost) for hallucination detection
- **Comprehensive Pipeline**: Complete workflow from data preprocessing to model training and evaluation
- **Interpretability Focus**: Uses interpretability techniques to understand model decisions
- **Modular Design**: Command-line scripts for easy automation and integration
- **Flexible Configuration**: Configurable parameters for all pipeline stages

## Project Structure

```
interpretablity-hallucination-detection/
├── datasets/                    # Data files
│   ├── OV_copying_score.json   # Overlap copying scores
│   ├── test/                   # Test dataset files
│   └── train/                  # Training dataset files
│       └── chunk_scores/       # Chunk-level scores for training
├── scripts/                    # Python scripts for pipeline execution
│   ├── preprocess.py           # Data preprocessing
│   ├── generate_response.py    # Response generation
│   ├── generate_labels.py      # Label generation
│   ├── filter.py               # Label filtering
│   ├── compute_scores.py       # Chunk-level score computation
│   ├── classifier.py           # Model training and evaluation
│   ├── predict.py              # Model prediction and evaluation
│   ├── helper.py               # Utility functions
│   └── baseline/               # Baseline implementations
│       ├── run_gpt.py          # GPT baseline
│       ├── run_groq.py         # Groq baseline
│       ├── run_hf.py           # HuggingFace baseline
│       ├── run_ragas.py        # RAGAS baseline
│       ├── run_refchecker.py   # RefChecker baseline
│       └── run_trulens.py      # TruLens baseline
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
cd interpretablity-hallucination-detection
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

### Command-Line Interface

All scripts support command-line arguments for flexible configuration. Use `--help` for detailed options:

```bash
python scripts/script_name.py --help
```

### Complete Pipeline Execution

#### 1. Data Preprocessing
```bash
python scripts/preprocess.py \
    --output_dir "../datasets" \
    --verbose
```

#### 2. Response Generation
```bash
python scripts/generate_response.py \
    --train_path "../datasets/train.jsonl" \
    --val_path "../datasets/val.jsonl" \
    --test_path "../datasets/test.jsonl" \
    --model_name "microsoft/DialoGPT-medium" \
    --output_dir "../datasets" \
    --max_tokens 1024 \
    --max_new_tokens 512 \
    --device "cuda" \
    --verbose
```

#### 3. Label Generation
```bash
python scripts/generate_labels.py \
    --train_path "../datasets/train_w_response.jsonl" \
    --val_path "../datasets/val_w_response.jsonl" \
    --test_path "../datasets/test_w_response.jsonl" \
    --output_dir "../datasets" \
    --lettuce_method "transformer" \
    --llm_client "openai" \
    --llm_models "gpt-4o" "gpt-4o-mini" \
    --verbose
```

#### 4. Label Filtering
```bash
python scripts/filter.py \
    --train_path "../datasets/train_w_labels.jsonl" \
    --val_path "../datasets/val_w_labels.jsonl" \
    --test_path "../datasets/test_w_labels.jsonl" \
    --output_dir "../datasets" \
    --llama_column "llama_judge" \
    --gpt_column "gpt_judge" \
    --use_confidence_threshold \
    --confidence_threshold 0.8 \
    --verbose
```

#### 5. Score Computation
```bash
python scripts/compute_scores.py \
    --input_path "../datasets/test/test1176_w_labels_filtered.jsonl" \
    --output_dir "../datasets/test" \
    --model_name "microsoft/DialoGPT-medium" \
    --hf_model_name "BAAI/bge-large-en-v1.5" \
    --device "cuda" \
    --batch_size 32 \
    --save_plots \
    --verbose
```

#### 6. Model Training
```bash
python scripts/classifier.py \
    --input_dir "../datasets/train/chunk_scores" \
    --output_dir "../trained_models" \
    --models "LogisticRegression" "RandomForest" "SVC" "XGBoost" \
    --test_size 0.2 \
    --balance_classes \
    --use_feature_selection \
    --save_plots \
    --verbose
```

#### 7. Model Prediction
```bash
python scripts/predict.py \
    --data_path "../datasets/test/test1176_w_chunk_score.json" \
    --model_path "../trained_models/model_XGBoost_3000.pickle" \
    --output_dir "../results" \
    --save_predictions \
    --save_plots \
    --verbose
```

### Baseline Comparisons

Compare your approach against various baselines:

#### GPT Baseline
```bash
python scripts/baseline/run_gpt.py \
    --data_path "../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json" \
    --models "gpt-4o" "gpt-4o-mini" \
    --save_results \
    --output_path "gpt_baseline_results.json"
```

#### Groq Baseline
```bash
python scripts/baseline/run_groq.py \
    --data_path "../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json" \
    --models "llama3-8b-8192" "llama3-70b-8192" \
    --save_results \
    --output_path "groq_baseline_results.json"
```

#### HuggingFace Baseline
```bash
python scripts/baseline/run_hf.py \
    --data_path "../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json" \
    --models "microsoft/DialoGPT-medium" "gpt2" \
    --device "cuda" \
    --save_results \
    --output_path "hf_baseline_results.json"
```

#### RAGAS Baseline
```bash
python scripts/baseline/run_ragas.py \
    --data_path "../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json" \
    --model "gpt-4o" \
    --save_results \
    --output_path "ragas_baseline_results.json"
```

#### RefChecker Baseline
```bash
python scripts/baseline/run_refchecker.py \
    --data_path "../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json" \
    --model "gpt-4o" \
    --batch_size 8 \
    --save_results \
    --output_path "refchecker_baseline_results.json"
```

#### TruLens Baseline
```bash
python scripts/baseline/run_trulens.py \
    --data_path "../datasets/test/test1176_w_chunk_score_gpt41mini_calibrated.json" \
    --model "gpt-4o" \
    --save_results \
    --output_path "trulens_baseline_results.json"
```

### Using Trained Models

The trained models can be loaded and used for inference:

```python
import pickle
import pandas as pd

# Load a trained model
with open('trained_models/model_XGBoost_3000.pickle', 'rb') as f:
    model = pickle.load(f)

# Prepare features (same format as training)
features = prepare_features(your_data)

# Make predictions
predictions = model.predict(features)
probabilities = model.predict_proba(features)
```

## Data Format

### Input Data
- **JSONL format** for training and test data
- Each entry contains: `question`, `context`, `response`, and `labels`
- For scored data: includes `scores` with attention and parameter knowledge scores

### Output Data
- **Chunk-level scores** for each response segment
- **Binary or continuous** hallucination scores
- **Model predictions** and confidence scores
- **Evaluation metrics** (precision, recall, F1-score)

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

### Script Parameters
All scripts support extensive parameter configuration:

- **Data paths**: Input/output file locations
- **Model selection**: Choose from available models
- **Processing options**: Batch sizes, device selection, etc.
- **Output control**: Save results, plots, verbose logging
- **Performance tuning**: Memory management, parallel processing

## Dependencies

The project uses the following key dependencies:

- **Core ML**: `scikit-learn`, `pandas`, `numpy`, `scipy`
- **Deep Learning**: `torch`, `transformers`, `sentence-transformers`
- **Interpretability**: `transformer-lens`
- **Evaluation**: `ragas`, `trulens`, `refchecker`
- **APIs**: `openai`, `groq`
- **Visualization**: `matplotlib`, `seaborn`

See `requirements.txt` for complete dependency list with version constraints.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **API Rate Limits**: Add delays between requests
3. **Missing Dependencies**: Install with `pip install -r requirements.txt`
4. **Model Loading Errors**: Check file paths and model compatibility

### Performance Tips

- Use GPU acceleration when available
- Adjust batch sizes based on available memory
- Use smaller models for faster inference
- Enable caching for repeated computations

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
- Contributors to the interpretability research community
- Users and testers of this framework
- The transformer-lens, ragas, and trulens communities for their excellent tools
