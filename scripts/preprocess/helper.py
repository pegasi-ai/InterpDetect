import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Sentence-Based Splitting (simple semantic units)
def get_sentence_spans(text):
    sentences = sent_tokenize(text)
    spans = []
    start = 0
    for sentence in sentences:
        start = text.find(sentence, start)
        end = start + len(sentence)
        spans.append((start, end))
        start = end
    return spans

# Clause-Based Splitting (more complex semantic units)
def split_clauses(text):
    matches = list(re.finditer(r'[^,;]+[,;]?', text))
    spans = [match.span() for match in matches if match.group().strip()]
    return spans

#Embedding-Based Semantic Segmentation (for deep semantics)
def split_text_semantic_chunks(text, model, similarity_threshold=0.75):
    import re

    # Step 1: Split into raw sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    embeddings = model.encode(sentences)

    chunks = []
    spans = []
    current_chunk = []
    current_start = 0

    for i, sent in enumerate(sentences):
        current_chunk.append(sent)
        if i == len(sentences) - 1 or cosine_similarity(
            [embeddings[i]], [embeddings[i + 1]])[0][0] < similarity_threshold:
            
            # Join the chunk and find start/end in original text
            chunk_text = " ".join(current_chunk)
            start_idx = text.find(current_chunk[0], current_start)
            end_idx = text.find(current_chunk[-1], start_idx) + len(current_chunk[-1])
            
            chunks.append(chunk_text)
            spans.append([start_idx, end_idx])
            current_start = end_idx
            current_chunk = []

    return spans


def clean_text(text):
    # Remove extra spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # Collapse multiple periods (e.g., ". . ." => ".")
    text = re.sub(r'\.{2,}', '.', text)

    # Fix spacing after punctuation
    text = re.sub(r'([.,!?;:])(?=\w)', r'\1 ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Capitalize first letter of each sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    return ' '.join(sentences)