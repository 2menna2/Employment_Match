#!/usr/bin/env python3
import json
import os
import logging
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ESCO_FILE_PATH = "data/esco_skills.json"
EMBEDDINGS_FILE_PATH = "data/esco_embeddings.npy"
BATCH_SIZE = 100

def load_esco_skills(file_path: str):
    """Load ESCO skills taxonomy."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                esco_skills = json.load(f)
                logger.info(f"Loaded {len(esco_skills)} skills from {file_path}")
                return esco_skills
        except Exception as e:
            logger.error(f"Failed to load ESCO file: {e}")
            return []
    logger.error(f"ESCO file {file_path} not found.")
    return []

def load_embedder():
    """Load sentence transformer model."""
    try:
        embedder = AutoModel.from_pretrained(EMBEDDER_MODEL)
        embedder_tokenizer = AutoTokenizer.from_pretrained(EMBEDDER_MODEL)
        logger.info(f"Successfully loaded embedder: {EMBEDDER_MODEL}")
        return embedder, embedder_tokenizer
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        return None, None

def get_embeddings(texts, embedder, embedder_tokenizer, batch_size=BATCH_SIZE):
    """Generate embeddings in batches."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            inputs = embedder_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                batch_embeddings = embedder(**inputs).last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
            logger.info(f"Processed batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
            return np.array([])
    return np.vstack(embeddings) if embeddings else np.array([])

def main():
    """Generate and save ESCO skill embeddings."""
    esco_skills = load_esco_skills(ESCO_FILE_PATH)
    if not esco_skills:
        logger.error("No skills loaded. Exiting.")
        return

    embedder, embedder_tokenizer = load_embedder()
    if embedder is None or embedder_tokenizer is None:
        logger.error("Failed to load embedder. Exiting.")
        return

    esco_texts = [skill["skill"] + ": " + skill["description"] for skill in esco_skills]
    embeddings = get_embeddings(esco_texts, embedder, embedder_tokenizer)
    if embeddings.size == 0:
        logger.error("No embeddings generated. Exiting.")
        return

    try:
        np.save(EMBEDDINGS_FILE_PATH, embeddings)
        logger.info(f"Saved embeddings to {EMBEDDINGS_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")

if __name__ == "__main__":
    main()