"""Detects a paragraph of text in an input image.

Example usage as a script:

  python fashion_aggregator/fashion_aggregator.py \
    "Two dogs playing in the snow"
"""
import os
import argparse
import pickle
from pathlib import Path
from typing import List, Any, Dict
from PIL import Image

from sentence_transformers import SentenceTransformer, util
import torch


STAGED_TEXT_ENCODER_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "text-encoder"
STAGED_IMG_ENCODER_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "img-encoder"
STAGED_IMG_EMBEDDINGS_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "img-embeddings"
RAW_PHOTOS_DIR = Path(__file__).resolve().parent / "data" / "photos"
MODEL_FILE = "model.pt"
EMBEDDINGS_FILE = "embeddings.pkl"


class TextEncoder:
    """Encodes the given text"""

    def __init__(self, model_path='clip-ViT-B-32-multilingual-v1'):
        if model_path is None:
            model_path = STAGED_TEXT_ENCODER_MODEL_DIRNAME / MODEL_FILE
        self.model = SentenceTransformer(model_path)

    @torch.no_grad()
    def encode(self, query: str) -> torch.Tensor:
        """Predict/infer text embedding for a given query."""
        query_emb = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        return query_emb


class ImageEnoder:
    """Encodes the given image"""

    def __init__(self, model_path='clip-ViT-B-32'):
        if model_path is None:
            model_path = STAGED_IMG_ENCODER_MODEL_DIRNAME / MODEL_FILE
        self.model = SentenceTransformer(model_path)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Predict/infer text embedding for a given query."""
        image_emb = self.model.encode([image], convert_to_tensor=True, show_progress_bar=False)
        return image_emb


class Retriever:
    """Retrieves relevant images for a given text embedding."""
    
    def __init__(self, image_embeddings_path=None):
        if image_embeddings_path is None:
            image_embeddings_path = STAGED_IMG_EMBEDDINGS_DIRNAME / EMBEDDINGS_FILE

        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEnoder()

        with open(image_embeddings_path, 'rb') as file:
            self.image_names, self.image_embeddings = pickle.load(file)  
        print("Images:", len(self.image_names))

    @torch.no_grad()
    def predict(self, text_query: str, k: int=10) -> List[Any]:
        """Return top-k relevant items for a given embedding"""
        query_emb = self.text_encoder.encode(text_query)
        relevant_images = util.semantic_search(query_emb, self.image_embeddings, top_k=k)[0]
        return relevant_images

    @torch.no_grad()
    def search_images(self, text_query: str, k: int=6) -> Dict[str, List[Any]]:
        """Return top-k relevant images for a given embedding"""
        images = self.predict(text_query, k)
        paths_and_scores = {"path": [], "score": []}
        for img in images:
            paths_and_scores["path"].append(os.path.join(RAW_PHOTOS_DIR, self.image_names[img["corpus_id"]]))
            paths_and_scores["score"].append(img["score"])
        return paths_and_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "text_query",
        type=str,
        help="Text query",
    )
    args = parser.parse_args()

    retriever = Retriever()
    print(f"Given query: {args.text_query}")
    print(retriever.predict(args.text_query))


if __name__ == "__main__":
    main()
