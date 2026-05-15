import numpy as np

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Abstract class for an object-to-text embedding model"""

    @abstractmethod
    def embed(self, *args, **kwargs) -> np.ndarray:
        pass


class TextEmbedding(EmbeddingModel):
    """Base wrapper for text-embedding models"""

    def __init__(self, path: str):
        self.model_path = path

    def embed(self, text: str, **kwargs) -> np.ndarray:
        raise NotImplementedError
