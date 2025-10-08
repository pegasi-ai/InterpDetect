# Preprocessing Pipeline

This folder contains scripts for preprocessing the RAGBench/FinQA dataset to generate training and test data for hallucination detection.

## Pipeline Overview

The preprocessing pipeline consists of four main steps:

```
1. Preprocess → 2. Generate Response → 3. Generate Labels → 4. Filter
```

### Step 1: Preprocess
**Script:** `preprocess.py`

Prepares the raw dataset for response generation.

```bash
python scripts/preprocess/preprocess.py \
  --output_dir ../../datasets
```

**Key Arguments:**
- `--output_dir`: Directory for preprocessed output

---

### Step 2: Generate Response

Generate responses using either HuggingFace models or OpenAI GPT models.

#### Option A: HuggingFace Models
**Script:** `generate_response_hf.py`

Use local HuggingFace models (e.g., Qwen, Llama) for response generation.

```bash
# Basic usage
python scripts/preprocess/generate_response_hf.py

# With custom parameters
python scripts/preprocess/generate_response_hf.py \
  --train_path datasets/train/train.jsonl \
  --test_path datasets/test/test.jsonl \
  --model_name "Qwen/Qwen3-0.6B" \
  --output_dir datasets \
  --train_samples 3000 \
  --test_samples 1176 \
  --max_new_tokens 1024 \
```

**Key Arguments:**
- `--model_name`: HuggingFace model name (default: `Qwen/Qwen3-0.6B`)
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`)
- `--max_new_tokens`: Maximum tokens to generate
- `--skip_train`: Skip training dataset processing
- `--skip_test`: Skip test dataset processing


#### Option B: OpenAI GPT Models
**Script:** `generate_response_gpt.py`

Use OpenAI API for response generation (requires API key).

```bash
# Basic usage
python scripts/preprocess/generate_response_gpt.py

# With custom parameters
python scripts/preprocess/generate_response_gpt.py \
  --train_path datasets/train/train.jsonl \
  --test_path datasets/test/test.jsonl \
  --model_name "gpt-4.1-mini" \
  --output_dir datasets \
  --train_samples 3000 \
  --test_samples 1176 \
  --max_new_tokens 1024 \
  --temperature 0.3
```

**Key Arguments:**
- `--model_name`: OpenAI model name (default: `gpt-4o-mini`)
- `--temperature`: Sampling temperature (default: 0.3)
- `--max_new_tokens`: Maximum tokens to generate

**Setup:**
1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

**Output:**
- Training: `datasets/train/train{N}_w_response_{model}.jsonl`
- Test: `datasets/test/test{N}_w_response_{model}.jsonl`

---

### Step 3: Generate Labels
**Script:** `generate_labels.py`

Generate hallucination labels using LettuceDetect and LLM-as-a-judge.

```bash
# Basic usage
python scripts/preprocess/generate_labels.py

# With custom parameters
python scripts/preprocess/generate_labels.py \
  --train_path datasets/train/train3000_w_response.jsonl \
  --test_path datasets/test/test1176_w_response.jsonl \
  --output_dir datasets \
  --llm_client "groq" \
  --llm_model "llama-3.1-70b-versatile" \
  --lettuce_method "transformer"
```

**Key Arguments:**
- `--lettuce_method`: LettuceDetect method (`transformer`, `embedding`, etc.)
- `--lettuce_model`: LettuceDetect model path
- `--llm_client`: LLM client for judge (`openai` or `groq`)
- `--llm_model`: LLM model name for judge evaluation

**Setup:**
- For OpenAI: Set `OPENAI_API_KEY` in environment or `.env`
- For Groq: Set `GROQ_API_KEY` in environment or `.env`

**Output:**
- Training: `datasets/train/train{N}_w_labels.jsonl`
- Test: `datasets/test/test{N}_w_labels.jsonl`

---

### Step 4: Filter
**Script:** `filter.py`

Filter datasets based on LLM judge agreement and confidence thresholds.

```bash
# Basic usage
python scripts/preprocess/filter.py

# With custom parameters
python scripts/preprocess/filter.py \
  --train_path datasets/train/train3000_w_labels.jsonl \
  --test_path datasets/test/test1176_w_labels.jsonl \
  --output_dir datasets \
  --use_confidence_threshold \
  --confidence_threshold 0.8
```

**Key Arguments:**
- `--use_confidence_threshold`: Enable confidence-based filtering
- `--confidence_threshold`: Minimum confidence score (0.0-1.0)
- `--skip_train`: Skip training dataset filtering
- `--skip_test`: Skip test dataset filtering

**Output:**
- Training: `datasets/train/train{N}_w_labels_filtered.jsonl`
- Test: `datasets/test/test{N}_w_labels_filtered.jsonl`

---

## Complete Pipeline Example

Run the entire pipeline from start to finish:

```bash
# Step 1: Preprocess (if needed)
python scripts/preprocess/preprocess.py

# Step 2: Generate responses using GPT
export OPENAI_API_KEY='your-key-here'
python scripts/preprocess/generate_response_gpt.py \
  --model_name "gpt-4.1-mini" \
  --train_samples 3000 \
  --test_samples 1176

# Step 3: Generate labels
export GROQ_API_KEY='your-groq-key-here'
python scripts/preprocess/generate_labels.py \
  --llm_client "groq" \
  --llm_model "llama-3.1-70b-versatile"

# Step 4: Filter datasets
python scripts/preprocess/filter.py \
  --use_confidence_threshold \
  --confidence_threshold 0.8
```

---

## Environment Setup

### Required Dependencies

```bash
# Install core dependencies
pip install -r ../../requirements.txt
```

### API Keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key-here
GROQ_API_KEY=your-groq-key-here
```

Or export them in your shell:

```bash
export OPENAI_API_KEY='your-openai-key-here'
export GROQ_API_KEY='your-groq-key-here'
```

---

## Output Files

After running the complete pipeline, you'll have:

```
datasets/
├── train/
│   ├── train.jsonl                           # Original data
│   ├── train3000_w_response.jsonl            # With responses
│   ├── train3000_w_labels.jsonl              # With labels
│   └── train3000_w_labels_filtered.jsonl     # Filtered (final)
└── test/
    ├── test.jsonl                            # Original data
    ├── test1176_w_response.jsonl             # With responses
    ├── test1176_w_labels.jsonl               # With labels
    └── test1176_w_labels_filtered.jsonl      # Filtered (final)
```

---

## Additional Scripts

### Helper Scripts
- `helper.py`: Utility functions for text processing and similarity computation

---

For more information, see the main project [README](../../README.md).
